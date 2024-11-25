import polars as pl
import polars.selectors as cs
import pytest
from nanuk import transform
from polars import testing


@pytest.fixture
def splits() -> dict[str, float]:
    return {"train": 0.5, "val": 0.25, "test": 0.25}


@pytest.fixture
def split():
    return pl.Series(
        [
            "train",
            "train",
            "train",
            "train",
            "train",
            "train",
            "val",
            "val",
            "test",
            "test",
        ]
    )


@pytest.fixture
def df(split) -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "split": split,
            "sample_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "time": [1, 2, 2, 3, 1, 1, 2, 3, 1, 2],
            "a": [1.0, 1.0, 1.0, 2.0, None, 3.0, 4.0, 5.0, 6.0, 7.0],
            "b": [None, 2, 3, 1, 2, 1, 3, 4, 5, 6],
            "c": [0, 0, 0, 0, 0, None, 1, 1, 1, 1],
        }
    )


def test_check_splits(splits):
    unnormalized_splits = {"train": 50.0, "val": 25.0, "test": 25.0}
    result = transform.check_splits(unnormalized_splits)
    assert result == splits
    expected_warning = "Split proportions were normalized to sum to 1.0: {'train': 0.5, 'val': 0.25, 'test': 0.25}"
    with pytest.warns(UserWarning, match=expected_warning):
        transform.assign_splits(unnormalized_splits, "sample_id")


def test_assign_splits(df: pl.LazyFrame, splits):
    result = transform.assign_splits(splits=splits, group="sample_id")
    testing.assert_frame_equal(df.select(result), df.select("split"))


def test_minmax_scale():
    data = pl.LazyFrame({"a": [-1.0, -0.5, 0.0, 1.0], "b": [2, 6, 10, 18]})
    expected = pl.LazyFrame({"a": [0.0, 0.25, 0.5, 1.0], "b": [0.0, 0.25, 0.5, 1.0]})
    result = data.with_columns(transform.minmax_scale(cs.numeric()))
    testing.assert_frame_equal(result, expected)


def test_zscore_scale():
    data = pl.LazyFrame({"a": [0, 0, 1, 1], "b": [0, 0, 1, 1]})
    expected = pl.LazyFrame({"a": [-1.0, -1.0, 1.0, 1.0], "b": [-1.0, -1.0, 1.0, 1.0]})
    result = data.with_columns(transform.zscore_scale(cs.numeric()))
    testing.assert_frame_equal(result, expected)


# @pytest.mark.parametrize("method", ["minmax", "zscore"])
def test_standardize():
    pass  # TODO: Implement test


# @pytest.mark.parametrize("method", ["mean", "median", "interpolate", "forword_fill"])
def test_impute():
    pass  # TODO: Implement test


def test_if_over():
    data = pl.LazyFrame({"a": [1, 2, 3, 4], "b": [1, 1, 2, 2]})
    expr = pl.col("a") * 2
    result = transform.if_over(expr, over=None)
    expected = data.with_columns(expr.alias("a"))
    testing.assert_frame_equal(data.with_columns(result), expected)
    result = transform.if_over(expr, over="b")
    expected = data.with_columns(expr.over("b").alias("a"))
    testing.assert_frame_equal(data.with_columns(result), expected)


def test_integration(df: pl.LazyFrame):
    splits = {"train": 0.5, "val": 0.25, "test": 0.25}
    assignment = transform.assign_splits(splits=splits, group="sample_id", name="split")
    df = df.with_columns(assignment)
    columns = cs.by_name("a", "b", "c")
    train = columns.filter(pl.col("split").eq("train"))
    imputation = transform.impute(columns, method="median", train=train)
    scale = transform.standardize(columns, method="zscore", train=train)
    transforms = [imputation, scale]
    df = transform.pipeline(df, transforms, over="time")
    pass
