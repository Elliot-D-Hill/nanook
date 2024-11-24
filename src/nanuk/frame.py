import polars as pl

from nanuk.typing import TFrame


def collect(df: TFrame) -> pl.DataFrame:
    match df:
        case pl.DataFrame():
            return df
        case pl.LazyFrame():
            return df.collect()
        case _:
            raise ValueError(f"Unsupported type: {type(df)}. Choose from: {TFrame}")


def get_columns(df: TFrame) -> list[str]:
    match df:
        case pl.DataFrame():
            return df.columns
        case pl.LazyFrame():
            return df.collect_schema().names()
        case _:
            raise ValueError(f"Unsupported type: {type(df)}. Choose from: {TFrame}")
