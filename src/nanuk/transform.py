import warnings

import polars as pl
import polars.selectors as cs
from polars._typing import IntoExpr

from nanuk.typing import Impute, Standardize


def check_splits(splits: dict[str, float]) -> dict[str, float]:
    split_sum = sum(splits.values())
    splits = {split: size / split_sum for split, size in splits.items()}
    if split_sum != 1.0:
        warnings.warn(f"Split proportions were normalized to sum to 1.0: {str(splits)}")
    return splits


def assign_splits(
    splits: dict[str, float], group: str | list[str], name: str = "split"
) -> pl.Expr:
    splits = check_splits(splits)
    lower = 0.0
    expr = pl.when(False).then(None)
    index = pl.col(group).rank(method="dense").sub(1)
    items = list(splits.items())
    for split, size in items[:-1]:
        upper = lower + size * index.max()
        expr = expr.when(index.is_between(lower, upper)).then(pl.lit(split))
        lower = upper
    last_split = items[-1][0]
    return expr.otherwise(pl.lit(last_split)).alias(name)


def safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    denominator = pl.when(denominator.eq(0)).then(1).otherwise(denominator)
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
    elif method == "interpolate":
        train = train.interpolate(method="linear")
    elif method == "forward_fill":
        train = train.forward_fill()
    else:
        raise ValueError(f"Unknown method: '{method}'. Choose from: {Impute}")
    return expr.fill_null(train)


def if_over(expr: pl.Expr, over: IntoExpr) -> pl.Expr:
    return pl.when(over is not None).then(expr.over(over)).otherwise(expr)


def pipeline[T: (pl.DataFrame, pl.LazyFrame)](
    df: T, transforms: list[pl.Expr], over: IntoExpr = None
) -> T:
    for transform in transforms:
        transform = if_over(expr=transform, over=over)
        df = df.with_columns(transform)
    return df
