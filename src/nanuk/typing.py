from typing import Literal, TypeAlias, TypeVar

import polars as pl

Standardize: TypeAlias = Literal["minmax", "zscore"]
Impute: TypeAlias = Literal["mean", "median", "interpolate", "ffill"]
TFrame = TypeVar("TFrame", pl.DataFrame, pl.LazyFrame)
