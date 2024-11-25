import polars as pl
import pytest
from nanuk import preprocess
from polars import testing


@pytest.fixture
def df() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "a": [1, 2, None, 4],
            "b": [None, None, None, 4],
            "c": [1, 1, 1, 1],
        }
    )


def test_drop_null_columns(df: pl.LazyFrame):
    result = preprocess.drop_null_columns(df=df, cutoff=0.5)
    testing.assert_frame_equal(result, df.select(["a", "c"]))


def test_filter_null_rows(df: pl.LazyFrame):
    result = preprocess.filter_null_rows(df=df, columns=pl.col("a", "b"))
    testing.assert_frame_equal(result, df.filter(pl.arange(pl.len()) != 2))


def test_drop_zero_variance(df: pl.LazyFrame):
    result = preprocess.drop_zero_variance(df=df, cutoff=0.5)
    testing.assert_frame_equal(result, df.select(["a"]))
