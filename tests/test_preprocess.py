import polars as pl
from nanuk.preprocess import drop_null_columns


def test_drop_null_columns():
    df = pl.LazyFrame(
        {
            "a": [1, 2, None, 4],
            "b": [None, None, None, 4],
            "c": [1, 2, 3, 4],
        }
    )
    result = drop_null_columns(df=df, cutoff=0.5).collect()
    expected_columns = ["a", "c"]
    assert result.columns == expected_columns


def test_filter_null_rows():
    pass
