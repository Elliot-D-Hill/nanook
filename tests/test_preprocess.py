import polars as pl
import pytest
from nanuk import preprocess
from polars import testing


@pytest.fixture
def lf() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "a": [1, 2, None, 4],
            "b": [None, None, None, 4],
            "c": [1, 1, 1, 1],
        }
    )


def test_drop_null_columns(lf: pl.LazyFrame):
    result = preprocess.drop_null_columns(frame=lf, cutoff=0.5)
    testing.assert_frame_equal(result, lf.select(["a", "c"]))


def test_filter_null_rows(lf: pl.LazyFrame):
    result = preprocess.filter_null_rows(frame=lf, columns=pl.col("a", "b"))
    testing.assert_frame_equal(result, lf.filter(pl.arange(pl.len()) != 2))


def test_drop_zero_variance(lf: pl.LazyFrame):
    result = preprocess.drop_zero_variance(frame=lf, cutoff=0.5)
    testing.assert_frame_equal(result, lf.select(["a"]))
