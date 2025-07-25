import warnings
from functools import reduce

import polars as pl
from polars import selectors as cs
from polars._typing import IntoExpr, JoinStrategy

from nanook.typing import Frame


def validate_splits(splits: dict[str, float]) -> dict[str, float]:
    split_sum = sum(splits.values())
    splits = {split: size / split_sum for split, size in splits.items()}
    if split_sum != 1.0:
        warnings.warn(f"Split proportions were normalized to sum to 1.0: {str(splits)}")
    return splits


def to_expr(value) -> pl.Expr:
    match value:
        case pl.Expr():
            return value
        case _:
            return pl.col(value)


def assign_splits[T: (pl.DataFrame, pl.LazyFrame)](
    frame: T,
    splits: dict[str, float],
    by: IntoExpr = None,
    stratify_by: IntoExpr = None,
    name: str = "split",
    shuffle: bool = True,
    seed: int | None = None,
) -> T:
    """
    Assigns splits to a DataFrame/LazyFrame based on the specified proportions and groupings.
    Args:
        frame: DataFrame/LazyFrame to assign splits to.
        splits: A dictionary of split names and their corresponding proportions.
        by: Column(s) to group by for assigning splits.
        stratify_by: Column(s) to stratify the splits.
        name: Name of the new column to store the assigned splits.
        shuffle: Whether to shuffle the groups before assigning splits.
        seed: Random seed for shuffling the groups.
    Returns:
        The input DataFrame/LazyFrame with the assigned splits as a new column.
    """
    splits = validate_splits(splits)
    split_list = list(splits.items())
    by = pl.int_range(pl.len()) if by is None else to_expr(by)
    group_id = pl.struct(by).rank(method="dense").sub(other=1)
    n_groups = group_id.n_unique()
    if shuffle:
        shuffled_id = pl.int_range(n_groups).shuffle(seed=seed)
        group_id = group_id.replace(group_id.unique(), shuffled_id)
    lower = pl.lit(0)
    expr = pl.when(False).then(None)
    for split, size in split_list[:-1]:
        upper = lower + (size * n_groups)
        assignment = group_id.is_between(lower, upper, closed="left")
        if stratify_by is not None:
            assignment = assignment.over(stratify_by)
        expr = expr.when(assignment).then(pl.lit(split))
        lower = upper
    last_split = pl.lit(split_list[-1][0])
    expr = expr.otherwise(last_split)
    return frame.select(expr.alias(name), cs.exclude(name))


def join_dataframes[T: (pl.DataFrame, pl.LazyFrame)](
    frames: list[T], on: str | list[str] | pl.Expr, how: JoinStrategy
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
