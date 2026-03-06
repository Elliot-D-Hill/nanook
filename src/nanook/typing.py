from typing import Literal, TypeAlias

Standardize: TypeAlias = Literal["minmax", "zscore"]
Impute: TypeAlias = Literal["mean", "median", "interpolate", "forword_fill"]
