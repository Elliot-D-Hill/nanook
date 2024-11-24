import warnings

import polars as pl
import polars.selectors as cs

from nanuk.typing import Impute, Standardize, TFrame


def check_splits(splits: dict[str, float]) -> dict[str, float]:
    split_sum = sum(splits.values())
    splits = {split: size / split_sum for split, size in splits.items()}
    if split_sum != 1.0:
        warnings.warn(f"Split proportions were normalized to sum to 1.0: {splits}")
    return splits


def assign_split(
    df: TFrame, splits: dict[str, float], group: str | list[str], name: str = "split"
) -> TFrame:
    splits = check_splits(splits)
    lower = 0.0
    expr = pl.when(False).then(None)
    index = pl.col(group).rank(method="dense").sub(1)
    for split, size in splits.items():
        upper = lower + size * index.max()
        expr = expr.when(index.is_between(lower, upper)).then(pl.lit(split))
        lower = upper
    return df.with_columns(expr.alias(name))


def if_over(expr: pl.Expr, over: str | list[str] | pl.Expr | None) -> pl.Expr:
    return pl.when(over is not None).then(expr.over(over)).otherwise(expr)


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


def standardize(expr: pl.Expr, method: str, train: pl.Expr = cs.numeric()) -> pl.Expr:
    if method == "minmax":
        expr = minmax_scale(expr=expr, train=train)
    elif method == "zscore":
        expr = zscore_scale(expr=expr, train=train)
    else:
        raise ValueError(f"Unknown method: '{method}'. Choose from: {Standardize}")
    return expr


def impute(expr: pl.Expr, method: str, train: pl.Expr = cs.numeric()) -> pl.Expr:
    if method == "mean":
        train = train.mean()
    elif method == "median":
        train = train.median()
    elif method == "interpolate":
        train = train.interpolate(method="linear")
    elif method == "ffill":
        train = train.forward_fill()
    else:
        raise ValueError(f"Unknown method: '{method}'. Choose from: {Impute}")
    return expr.fill_null(train)


def pipeline(df: TFrame, transforms: list[pl.Expr], over: str | None = None) -> TFrame:
    for transform in transforms:
        transform = if_over(expr=transform, over=over)
        df = df.with_columns(transform)
    return df
