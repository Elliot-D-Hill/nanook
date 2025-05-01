import polars as pl
import polars.selectors as cs

from nanook.frame import collect_if_lazy


def select_by_condition[T: (pl.DataFrame, pl.LazyFrame)](
    frame: T, condition: pl.Expr
) -> T:
    condition_df = collect_if_lazy(frame.select(condition).unpivot())
    columns_to_keep = condition_df.filter(pl.col("value"))["variable"]
    return frame.select(cs.by_name(columns_to_keep))


def drop_null_columns[T: (pl.DataFrame, pl.LazyFrame)](frame: T, cutoff: float) -> T:
    is_null = pl.all().null_count().truediv(pl.len()).lt(cutoff)
    return select_by_condition(frame=frame, condition=is_null)


def filter_null_rows[T: (pl.DataFrame, pl.LazyFrame)](frame: T, columns: pl.Expr) -> T:
    return frame.filter(~pl.all_horizontal(columns.is_null()))


def drop_low_variance[T: (pl.DataFrame, pl.LazyFrame)](
    frame: T, cutoff: float = 1e-8
) -> T:
    has_zero_variance = pl.all().var().gt(cutoff)
    return select_by_condition(frame=frame, condition=has_zero_variance)
