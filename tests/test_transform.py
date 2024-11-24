import polars as pl
import polars.selectors as cs
import pytest
from nanuk import transform
from nanuk.typing import Frame
from polars import testing


@pytest.fixture
def splits():
    return {"train": 0.5, "val": 0.25, "test": 0.25}


@pytest.fixture
def unnormalized_splits():
    return {"train": 50.0, "val": 25.0, "test": 25.0}


@pytest.fixture
def df() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "split": [
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
            ],
            "sample_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "time": [1, 2, 2, 3, 1, 1, 2, 3, 1, 2],
            "a": [1.0, 1.0, 1.0, 2.0, None, 3.0, 4.0, 5.0, 6.0, 7.0],
            "b": [None, 2, 3, 1, 2, 1, 3, 4, 5, 6],
            "c": [0, 0, 0, 0, 0, None, 1, 1, 1, 1],
        }
    )


def test_check_splits(
    df: Frame, splits: dict[str, float], unnormalized_splits: dict[str, float]
):
    result = transform.check_splits(unnormalized_splits)
    assert result == splits
    expected_warning = "Split proportions were normalized to sum to 1.0: {'train': 0.5, 'val': 0.25, 'test': 0.25}"
    with pytest.warns(UserWarning, match=expected_warning):
        transform.assign_split(df, unnormalized_splits, "sample_id")


def test_assign_splits(df: Frame, splits: dict[str, float]):
    result = transform.assign_split(df=df, splits=splits, group="sample_id")
    testing.assert_frame_equal(result.select("split"), df.select("split"))


def test_minmax_scale():
    schema = ["a", "b"]
    data = pl.DataFrame([[-1, 2], [-0.5, 6], [0, 10], [1, 18]], schema=schema)
    expected = pl.DataFrame(
        [[0.0, 0.0], [0.25, 0.25], [0.5, 0.5], [1.0, 1.0]], schema=schema
    )
    result = data.with_columns(transform.minmax_scale(cs.numeric()))
    testing.assert_frame_equal(result, expected)


def test_zscore_scale():
    schema = ["a", "b"]
    data = pl.DataFrame([[0, 0], [0, 0], [1, 1], [1, 1]], schema=schema)
    expected = pl.DataFrame(
        [[-1.0, -1.0], [-1.0, -1.0], [1.0, 1.0], [1.0, 1.0]], schema=schema
    )
    result = data.with_columns(transform.zscore_scale(cs.numeric()))
    testing.assert_frame_equal(result, expected)


def test_standardize():
    pass


def test_impute():
    pass


def test_if_over():
    pass


def test_integration(df: Frame, splits: dict[str, float]):
    df = transform.assign_split(df, splits=splits, group="sample_id", name="split")
    columns = cs.numeric()
    train = columns.filter(pl.col("split").eq("train"))
    scale = transform.standardize(columns, method="zscore", train=train)
    imputation = transform.impute(columns, method="median", train=train)
    transforms = [scale, imputation]
    df = transform.pipeline(df, transforms, over="time")
    print(df)
