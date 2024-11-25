import polars as pl

from nanuk.typing import Frame


def collect_if_lazy(df: Frame) -> pl.DataFrame:
    match df:
        case pl.DataFrame():
            return df
        case pl.LazyFrame():
            return df.collect()
        case _:
            raise ValueError(f"Unsupported type: {type(df)}. Choose from: {Frame}")


def get_columns(df: Frame) -> list[str]:
    match df:
        case pl.DataFrame():
            return df.columns
        case pl.LazyFrame():
            return df.collect_schema().names()
        case _:
            raise ValueError(f"Unsupported type: {type(df)}. Choose from: {Frame}")
