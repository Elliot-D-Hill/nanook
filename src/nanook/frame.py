from functools import reduce

import polars as pl
from polars._typing import JoinStrategy


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
