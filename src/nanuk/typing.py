from typing import Literal, TypeAlias

import polars as pl

Standardize: TypeAlias = Literal["minmax", "zscore"]
Impute: TypeAlias = Literal["mean", "median", "interpolate", "forword_fill"]
type Frame = pl.DataFrame | pl.LazyFrame
