import warnings
from functools import reduce
from typing import Iterable

import polars as pl
from polars import selectors as cs
from polars._typing import JoinStrategy

from nanook.typing import Frame


def validate_splits(splits: dict[str, float]) -> dict[str, float]:
    split_sum = sum(splits.values())
    splits = {split: size / split_sum for split, size in splits.items()}
    if split_sum != 1.0:
        warnings.warn(f"Split proportions were normalized to sum to 1.0: {str(splits)}")
    return splits


def to_expr(value: str | Iterable[str] | pl.Expr) -> pl.Expr:
    match value:
        case str() | list():
            return pl.col(value)
        case pl.Expr():
            return value
        case _:
            raise ValueError(f"Unsupported type for 'value': {type(value)}.")


def assign_splits[T: (pl.DataFrame, pl.LazyFrame)](
    frame: T,
    splits: dict[str, float],
    by: str | list[str] | pl.Expr | None = None,
    stratify_by: str | list[str] | pl.Expr | None = None,
    name: str = "split",
) -> T:
    """
    Assigns splits to a DataFrame/LazyFrame based on the specified proportions and groupings.
    Args:
        frame: DataFrame/LazyFrame to assign splits to.
        splits: A dictionary of split names and their corresponding proportions.
        by: Column(s) to group by for assigning splits.
        stratify_by: Column(s) to stratify the splits.
        name: Name of the new column to store the assigned splits.
    Returns:
        The input DataFrame/LazyFrame with the assigned splits as a new column.
    """
    splits = validate_splits(splits)
    split_list = list(splits.items())
    by = to_expr(by) if by is not None else pl.int_range(pl.len())
    index = by.rank(method="dense").sub(other=1)
    expr = pl.when(False).then(None)
    lower = 0.0
    for split, size in split_list[:-1]:
        upper = lower + size * index.max()
        assignment = index.is_between(lower, upper).over(stratify_by)
        expr = expr.when(assignment).then(pl.lit(split))
        lower = upper
    expr = expr.otherwise(pl.lit(split_list[-1][0]))
    return frame.select(expr.alias(name), cs.exclude(name))


def join_dataframes[T: (pl.DataFrame, pl.LazyFrame)](
    frames: list[T], on: str | list[str], how: JoinStrategy
) -> T:
    return reduce(
        lambda left, right: left.join(right, how=how, coalesce=True, on=on), frames
    )


def collect_if_lazy(frame: Frame) -> pl.DataFrame:
    match frame:
        case pl.DataFrame():
            return frame
        case pl.LazyFrame():
            return frame.collect()
        case _:
            raise ValueError(f"Unsupported type: {type(frame)}.")


def get_column_names(frame: Frame) -> list[str]:
    match frame:
        case pl.DataFrame():
            return frame.columns
        case pl.LazyFrame():
            return frame.collect_schema().names()
        case _:
            raise ValueError(f"Unsupported type: {type(frame)}.")
