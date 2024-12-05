import polars as pl
import polars.selectors as cs
import pytest
from nanuk import transform
from polars import testing


@pytest.fixture
def splits() -> dict[str, float]:
    return {"a": 0.5, "b": 0.25, "c": 0.25}


@pytest.fixture
def lf() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "split": ["a", "a", "a", "a", "a", "a", "b", "b", "c", "c"],
            "id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "time": [1, 2, 2, 3, 1, 1, 2, 3, 1, 2],
            "a": [1.0, 1.0, 1.0, 2.0, None, 3.0, 4.0, 5.0, 6.0, 7.0],
            "b": [None, 2, 3, 1, 2, 1, 3, 4, 5, 6],
            "c": [0, 0, 0, 0, 0, None, 1, 1, 1, 1],
        }
    )


@pytest.fixture
def lf_splits():
    return pl.LazyFrame(
        {
            "split": ["a", "a", "b", "c", "a", "a", "b", "c"],
            "time": [1, 2, 3, 4, 1, 2, 3, 4],
            "id": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )


def test_check_splits(lf: pl.LazyFrame, splits: dict[str, float]):
    unnormalized_splits = {"a": 50.0, "b": 25.0, "c": 25.0}
    result = transform.check_splits(unnormalized_splits)
    assert result == splits
    expected_warning = "Split proportions were normalized to sum to 1.0: {'a': 0.5, 'b': 0.25, 'c': 0.25}"
    with pytest.warns(UserWarning, match=expected_warning):
        transform.assign_splits(lf, splits=unnormalized_splits, by="id")


def test_assign_splits(lf: pl.LazyFrame, splits: dict[str, float]):
    result = transform.assign_splits(frame=lf, splits=splits, by="id")
    testing.assert_frame_equal(result, lf)


def test_assign_splits_over(lf_splits: pl.LazyFrame):
    splits = {"a": 0.6, "b": 0.2, "c": 0.2}
    result = transform.assign_splits(
        frame=lf_splits, splits=splits, by="time", over="id", name="split"
    )
    testing.assert_frame_equal(result, lf_splits, check_row_order=False)
    df = lf_splits.collect().sample(fraction=1, with_replacement=False, shuffle=True)
    result = transform.assign_splits(
        frame=df, splits=splits, by="time", over="id", name="split"
    )
    testing.assert_frame_equal(result, lf_splits.collect(), check_row_order=False)


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


# def test_integration(lf: pl.LazyFrame):
#     splits = {"train": 0.5, "val": 0.25, "test": 0.25}
#     lf = transform.assign_splits(lf, splits=splits, by="id", name="split")
#     columns = cs.by_name("a", "b", "c")
#     train = columns.filter(pl.col("split").eq("train"))
#     imputation = transform.impute(columns, method="median", train=train)
#     scale = transform.standardize(columns, method="zscore", train=train)
#     transforms = [imputation, scale]
#     lf = transform.pipeline(lf, transforms, over="time")
#     pass
