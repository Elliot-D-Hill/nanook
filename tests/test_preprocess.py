import polars as pl
import pytest
from polars import testing

from nanook import preprocess


@pytest.fixture
def preprocess_lf() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "a": [1, 2, None, 4],
            "b": [None, None, None, 4],
            "c": [1, 1, 1, 1],
        }
    )


def test_drop_null_columns(preprocess_lf: pl.LazyFrame):
    result = preprocess.drop_null_columns(frame=preprocess_lf, cutoff=0.5)
    testing.assert_frame_equal(result, preprocess_lf.select(["a", "c"]))


def test_filter_null_rows(preprocess_lf: pl.LazyFrame):
    result = preprocess.filter_null_rows(frame=preprocess_lf, columns=pl.col("a", "b"))
    testing.assert_frame_equal(result, preprocess_lf.filter(pl.arange(pl.len()) != 2))


def test_drop_low_variance(preprocess_lf: pl.LazyFrame):
    result = preprocess.drop_low_variance(frame=preprocess_lf, cutoff=0.5)
    testing.assert_frame_equal(result, preprocess_lf.select(["a"]))
