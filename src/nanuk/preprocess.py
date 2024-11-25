import polars as pl

from nanuk.frame import collect_if_lazy


def drop_null_columns[T: (pl.DataFrame, pl.LazyFrame)](df: T, cutoff: float) -> T:
    n_rows = df.select(pl.len())
    n_rows = collect_if_lazy(n_rows).item()
    column_is_null = pl.all().null_count().truediv(n_rows).lt(cutoff)
    columns = collect_if_lazy(df.select(column_is_null)).to_dicts()[0]
    columns_to_keep = [col for col, is_null in columns.items() if is_null]
    return df.select(columns_to_keep)


def filter_null_rows[T: (pl.DataFrame, pl.LazyFrame)](df: T, columns: pl.Expr) -> T:
    return df.filter(~pl.all_horizontal(columns.is_null()))


def drop_zero_variance[T: (pl.DataFrame, pl.LazyFrame)](df: T, cutoff: float) -> T:
    return df.select(pl.all().var().gt(cutoff))
