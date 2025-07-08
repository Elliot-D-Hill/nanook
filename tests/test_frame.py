from typing import cast

import polars as pl
import pytest
from polars import testing

from nanook import frame


def test_validate_splits(splits: dict[str, float]):
    unnormalized_splits = {"a": 50.0, "b": 25.0, "c": 25.0}
    expected_warning = "Split proportions were normalized to sum to 1.0: {'a': 0.5, 'b': 0.25, 'c': 0.25}"
    with pytest.warns(UserWarning, match=expected_warning):
        result = frame.validate_splits(unnormalized_splits)
    assert result == splits


def test_to_expr():
    assert isinstance(frame.to_expr("a"), pl.Expr)
    assert isinstance(frame.to_expr(["a", "b"]), pl.Expr)
    assert isinstance(frame.to_expr(pl.col("a")), pl.Expr)


def test_assign_splits(splits: dict[str, float]):
    df = pl.DataFrame({"id": [0, 0, 1, 1, 2, 2, 3, 3]})
    df = frame.assign_splits(frame=df, splits=splits, by="id", seed=42)
    expected = df.group_by("split").agg(pl.len().truediv(df.height).alias("frac"))
    for split, frac in splits.items():
        assert expected.filter(pl.col("split") == split)["frac"].item() == frac


def test_join_dataframes():
    df1 = pl.DataFrame({"id": [1, 2, 3], "val1": ["a", "b", "c"]})
    df2 = pl.DataFrame({"id": [2, 3, 4], "val2": [10, 20, 30]})
    df3 = pl.DataFrame({"id": [3, 4, 5], "val3": [True, False, True]})
    expected_inner = pl.DataFrame(
        {"id": [3], "val1": ["c"], "val2": [20], "val3": [True]}
    )
    result_inner = frame.join_dataframes([df1, df2, df3], on="id", how="inner")
    testing.assert_frame_equal(
        result_inner.sort("id"), expected_inner.sort("id"), check_column_order=False
    )
    lf1 = df1.lazy()
    lf2 = df2.lazy()
    expected_left_data = {
        "id": [1, 2, 3],
        "val1": ["a", "b", "c"],
        "val2": [None, 10, 20],
    }
    expected_left = pl.LazyFrame(expected_left_data)
    result_left_lf = frame.join_dataframes([lf1, lf2], on="id", how="left")
    testing.assert_frame_equal(
        result_left_lf.collect().sort("id"),
        expected_left.collect().sort("id"),
        check_column_order=False,
    )


def test_collect_if_lazy():
    df_input = pl.DataFrame({"a": [1]})
    result_df = frame.collect_if_lazy(df_input)
    assert isinstance(result_df, pl.DataFrame)
    testing.assert_frame_equal(result_df, df_input)
    lf_input = pl.LazyFrame({"b": [2]})
    result_collected_lf = frame.collect_if_lazy(lf_input)
    assert isinstance(result_collected_lf, pl.DataFrame)
    testing.assert_frame_equal(result_collected_lf, lf_input.collect())


def test_get_column_names():
    df = pl.DataFrame({"x": [1], "y": ["a"]})
    assert frame.get_column_names(df) == ["x", "y"]
    lf = pl.LazyFrame({"x": [1.0], "y": [True]})
    assert frame.get_column_names(lf) == ["x", "y"]


def assert_props_close(
    original_df: pl.DataFrame, split_df: pl.DataFrame, by, tolerance=0.05
):
    original_df = (
        original_df.group_by(by)
        .agg(pl.len().alias("orig_count"))
        .with_columns((pl.col("orig_count") / original_df.height).alias("orig_prop"))
    )
    split_dataframe = (
        split_df.group_by(by)
        .agg(pl.len().alias("split_count"))
        .with_columns((pl.col("split_count") / split_df.height).alias("split_prop"))
    )
    comparison = original_df.join(split_dataframe, on=by, how="inner").with_columns(
        (pl.col("orig_prop") - pl.col("split_prop")).abs().alias("prop_diff")
    )
    max_difference = comparison["prop_diff"].max()
    max_difference = cast(float, max_difference)
    assert max_difference < tolerance, (
        f"Max proportion diff {comparison['prop_diff'].max()} ≥ {tolerance}"
    )


def test_stratify_by_column(df: pl.DataFrame):
    out = frame.assign_splits(
        df, splits={"train": 0.7, "test": 0.3}, stratify_by="category", seed=42
    )
    train = out.filter(pl.col("split") == "train")
    test = out.filter(pl.col("split") == "test")
    assert_props_close(df, train, by="category", tolerance=0.05)
    assert_props_close(df, test, by="category", tolerance=0.05)


def test_stratify_by_column_repeated():
    df = pl.DataFrame(
        {
            "split": ["a", "a", "b", "b", "a", "a", "a", "a", "b", "b", "b", "b"],
            "id": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "label": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        }
    )
    splits = {"a": 0.5, "b": 0.5}
    result = frame.assign_splits(df, splits=splits, stratify_by="label", shuffle=False)
    testing.assert_series_equal(
        left=result["split"], right=df["split"], check_dtypes=False
    )


def test_stratify_by_list_of_columns(df: pl.DataFrame):
    df = df.with_columns(
        (pl.col("category") + (pl.col("value") > 0).cast(pl.Utf8)).alias("combo")
    )
    out = frame.assign_splits(
        df, splits={"train": 0.5, "eval": 0.5}, stratify_by=["category", "combo"]
    )
    for split_name in ["train", "eval"]:
        part = out.filter(pl.col("split") == split_name)
        orig_counts = df.group_by(["category", "combo"]).agg(pl.len().alias("orig_n"))
        split_counts = part.group_by(["category", "combo"]).agg(pl.len())
        joined = orig_counts.join(split_counts, on=["category", "combo"], how="left")
        missing = joined.filter(pl.col("len").is_null())
        assert missing.height == 0, f"Missing groups in split {split_name}:\n{missing}"


def test_assign_splits_disjoint_by_groups():
    """Test that when using 'by', groups are disjoint across splits."""
    df = pl.DataFrame(
        {
            "id": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        }
    )
    result = frame.assign_splits(
        df, splits={"train": 0.6, "val": 0.2, "test": 0.2}, by="id", seed=42
    )

    train_ids = set(result.filter(pl.col("split") == "train")["id"].unique().to_list())
    val_ids = set(result.filter(pl.col("split") == "val")["id"].unique().to_list())
    test_ids = set(result.filter(pl.col("split") == "test")["id"].unique().to_list())

    # Verify no overlap between splits
    assert len(train_ids & val_ids) == 0, (
        "Train and validation splits have overlapping IDs"
    )
    assert len(train_ids & test_ids) == 0, "Train and test splits have overlapping IDs"
    assert len(val_ids & test_ids) == 0, (
        "Validation and test splits have overlapping IDs"
    )
    # Verify all original IDs are present
    all_split_ids = train_ids | val_ids | test_ids
    original_ids = set(df["id"].unique().to_list())
    assert all_split_ids == original_ids, "Some IDs were lost during splitting"
