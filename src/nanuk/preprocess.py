import polars as pl
import polars.selectors as cs

from nanuk.frame import collect_if_lazy


def select_by_condition[T: (pl.DataFrame, pl.LazyFrame)](
    df: T, condition: pl.Expr
) -> T:
    condition_df = collect_if_lazy(df.select(condition).unpivot())
    columns_to_keep = condition_df.filter(pl.col("value"))["variable"]
    return df.select(cs.by_name(columns_to_keep))


def drop_null_columns[T: (pl.DataFrame, pl.LazyFrame)](df: T, cutoff: float) -> T:
    is_null = pl.all().null_count().truediv(pl.len()).lt(cutoff)
    return select_by_condition(df=df, condition=is_null)


def filter_null_rows[T: (pl.DataFrame, pl.LazyFrame)](df: T, columns: pl.Expr) -> T:
    return df.filter(~pl.all_horizontal(columns.is_null()))


def drop_zero_variance[T: (pl.DataFrame, pl.LazyFrame)](
    df: T, cutoff: float = 1e-8
) -> T:
    has_zero_variance = pl.all().var().gt(cutoff)
    return select_by_condition(df=df, condition=has_zero_variance)
