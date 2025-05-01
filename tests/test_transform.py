import polars as pl
import polars.selectors as cs
from polars import testing

from nanook import transform


def test_minmax_scale():
    data = pl.LazyFrame({"a": [-1.0, -0.5, 0.0, 1.0], "b": [2, 6, 10, 18]})
    expected = pl.LazyFrame({"a": [0.0, 0.25, 0.5, 1.0], "b": [0.0, 0.25, 0.5, 1.0]})
    result = data.with_columns(transform.minmax_scale(cs.numeric()))
    testing.assert_frame_equal(result, expected)


def test_zscore_scale():
    data = pl.LazyFrame({"a": [0, 0, 1, 1], "b": [0, 0, 1, 1]})
    expected = pl.LazyFrame({"a": [-1.0, -1.0, 1.0, 1.0], "b": [-1.0, -1.0, 1.0, 1.0]})
    result = data.with_columns(transform.zscore_scale(cs.numeric()))
    testing.assert_frame_equal(result, expected)


# @pytest.mark.parametrize("method", ["minmax", "zscore"])
def test_standardize():
    pass  # TODO: Implement test


# @pytest.mark.parametrize("method", ["mean", "median", "interpolate", "forword_fill"])
def test_impute():
    pass  # TODO: Implement test


# def test_integration(lf: pl.LazyFrame):
#     splits = {"train": 0.5, "val": 0.25, "test": 0.25}
#     lf = transform.assign_splits(lf, splits=splits, by="id", name="split")
#     columns = cs.by_name("a", "b", "c")
#     train = columns.filter(pl.col("split").eq("train"))
#     imputation = transform.impute(columns, method="median", train=train)
#     scale = transform.standardize(columns, method="zscore", train=train)
#     transforms = [imputation, scale]
#     lf = transform.pipeline(lf, transforms, over="time")
#     pass
