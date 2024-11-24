import polars as pl

from nanuk.frame import collect_if_lazy
from nanuk.typing import TFrame


def drop_null_columns(df: TFrame, cutoff: float) -> TFrame:
    n_rows = df.select(pl.len())
    n_rows = collect_if_lazy(n_rows).item()
    column_is_null = pl.all().null_count().truediv(n_rows).lt(cutoff)
    columns = collect_if_lazy(df.select(column_is_null)).to_dicts()[0]
    columns_to_keep = [col for col, is_null in columns.items() if is_null]
    return df.select(columns_to_keep)


def filter_null_rows(df: TFrame, columns: pl.Expr) -> TFrame:
    return df.filter(~pl.all_horizontal(columns.is_null()))


def drop_zero_variance(df: TFrame, cutoff: float) -> TFrame:
    return df.select(pl.all().var().gt(cutoff))
