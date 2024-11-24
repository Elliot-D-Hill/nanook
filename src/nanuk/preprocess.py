import polars as pl

Frame = pl.DataFrame | pl.LazyFrame


def drop_null_columns(df: pl.LazyFrame, cutoff: float) -> pl.LazyFrame:
    n_rows = df.select(pl.len()).collect().item()
    null_proportion = df.null_count().with_columns(pl.all().truediv(n_rows))
    keep = null_proportion.select(pl.all().lt(cutoff)).collect()
    kept_columns = [col for col in df.columns if keep[col].item()]
    return df.select(kept_columns)


# TODO make cutoff parameter
def filter_null_rows(df: pl.LazyFrame, columns: pl.Expr) -> pl.LazyFrame:
    return df.filter(~pl.all_horizontal(columns.is_null()))


def drop_zero_variance(df: pl.LazyFrame, cutoff: float) -> pl.LazyFrame:
    to_drop = df.select(pl.all().var().lt(cutoff)).collect()
    to_drop = [col for col in to_drop.columns if to_drop[col].item()]
    return df.drop(to_drop)
