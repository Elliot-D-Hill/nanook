from typing import Literal, TypeVar

import polars as pl

type Standardize = Literal["minmax", "zscore"]
type Impute = Literal["mean", "median", "interpolate", "ffill"]
TFrame = TypeVar("TFrame", pl.DataFrame, pl.LazyFrame)
