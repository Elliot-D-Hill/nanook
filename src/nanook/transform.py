import polars as pl
import polars.selectors as cs
from polars._typing import IntoExpr

from nanook.typing import Impute, Standardize


def identity[T](value: T) -> T:
    return value


def safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    denominator = pl.when(denominator.eq(0.0)).then(1.0).otherwise(denominator)
    return numerator.truediv(denominator)


def minmax_scale(expr: pl.Expr, train: pl.Expr = cs.numeric()) -> pl.Expr:
    minimum = train.min()
    maximum = train.max()
    difference = maximum - minimum
    return safe_divide(expr.sub(minimum), difference)


def zscore_scale(expr: pl.Expr, train: pl.Expr = cs.numeric()) -> pl.Expr:
    return safe_divide(expr.sub(train.mean()), train.std(ddof=0))


def standardize(expr: pl.Expr, method: str, train: pl.Expr | None = None) -> pl.Expr:
    train = expr if train is None else train
    if method == "minmax":
        expr = minmax_scale(expr=expr, train=train)
    elif method == "zscore":
        expr = zscore_scale(expr=expr, train=train)
    else:
        raise ValueError(f"Unknown method: '{method}'. Choose from: {Standardize}")
    return expr


def impute(expr: pl.Expr, method: str, train: pl.Expr | None = None) -> pl.Expr:
    train = expr if train is None else train
    if method == "mean":
        train = train.mean()
    elif method == "median":
        train = train.median()
    else:
        raise ValueError(f"Unknown method: '{method}'. Choose from: {Impute}")
    return expr.fill_null(train)


def pipeline[T: (pl.DataFrame, pl.LazyFrame)](
    frame: T, transforms: list[pl.Expr], over: IntoExpr | None = None
) -> T:
    for transform in transforms:
        frame = frame.with_columns(transform.over(over))
    return frame
