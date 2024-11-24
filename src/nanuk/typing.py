from typing import Literal, TypeAlias

import polars as pl

Standardize: TypeAlias = Literal["minmax", "zscore"]
Impute: TypeAlias = Literal["mean", "median", "interpolate", "ffill"]
Frame: TypeAlias = pl.DataFrame | pl.LazyFrame
