import polars as pl
import pytest
from nanuk import preprocess


@pytest.fixture
def df() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "a": [1, 2, None, 4],
            "b": [None, None, None, 4],
            "c": [1, 2, 3, 4],
        }
    )


def test_drop_null_columns(df: pl.LazyFrame):
    result = preprocess.drop_null_columns(df=df, cutoff=0.5)
    expected_columns = ["a", "c"]
    assert result.columns == expected_columns


def test_filter_null_rows(df: pl.LazyFrame):
    result = preprocess.filter_null_rows(df=df, columns=pl.col("a", "b"))
    expected_rows = 3
    assert result.collect().height == expected_rows
