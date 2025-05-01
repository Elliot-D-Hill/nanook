import warnings
from functools import reduce

import polars as pl
from polars import selectors as cs
from polars._typing import JoinStrategy


def validate_splits(splits: dict[str, float]) -> dict[str, float]:
    split_sum = sum(splits.values())
    splits = {split: size / split_sum for split, size in splits.items()}
    if split_sum != 1.0:
        warnings.warn(f"Split proportions were normalized to sum to 1.0: {str(splits)}")
    return splits


def assign_splits[T: (pl.DataFrame, pl.LazyFrame)](
    frame: T,
    splits: dict[str, float],
    by: str | list[str],
    name: str = "split",
) -> T:
    """
    Assigns splits to a DataFrame/LazyFrame based on the specified proportions and groupings.
    Args:
        frame: The DataFrame/LazyFrame to assign splits to.
        splits: A dictionary of split names and their corresponding proportions.
        by: The column(s) to group by for assigning splits.
        name: The name of the new column to store the assigned splits.
    Returns:
        A DataFrame/LazyFrame with the assigned splits.
    """
    splits = validate_splits(splits)
    splits = list(splits.items())
    groups = pl.col(by)
    frame = frame.sort(groups)
    index = groups.rank(method="dense").sub(other=1)
    max_index = index.max()
    expr = pl.when(False).then(None)
    lower = 0.0
    for split, size in splits[:-1]:
        upper = lower + size * max_index
        expr = expr.when(index.is_between(lower, upper)).then(pl.lit(split))
        lower = upper
    expr = expr.otherwise(pl.lit(splits[-1][0]))
    return frame.select(expr.alias(name), cs.exclude(name))


def join_dataframes[T: (pl.DataFrame, pl.LazyFrame)](
    frames: list[T], on: str | list[str], how: JoinStrategy
) -> T:
    return reduce(
        lambda left, right: left.join(right, how=how, coalesce=True, on=on), frames
    )


def collect_if_lazy[T: (pl.DataFrame, pl.LazyFrame)](frame: T) -> pl.DataFrame:
    match frame:
        case pl.DataFrame():
            return frame
        case pl.LazyFrame():
            return frame.collect()
        case _:
            raise ValueError(f"Unsupported type: {type(frame)}.")


def get_column_names[T: (pl.DataFrame, pl.LazyFrame)](frame: T) -> list[str]:
    match frame:
        case pl.DataFrame():
            return frame.columns
        case pl.LazyFrame():
            return frame.collect_schema().names()
        case _:
            raise ValueError(f"Unsupported type: {type(frame)}.")
