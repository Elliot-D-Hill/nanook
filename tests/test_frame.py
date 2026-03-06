from typing import cast

import polars as pl
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from polars import testing
from polars.testing import assert_frame_equal

from nanook import frame
from nanook.frame import lazy_sample


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


# ── Hypothesis strategies ─────────────────────────────────────────────────────


@st.composite
def dataframes(draw, min_rows=1, max_rows=30):
    """Generate small Polars DataFrames with integer columns."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_cols = draw(st.integers(min_value=1, max_value=4))
    data = {
        f"col_{i}": draw(
            st.lists(st.integers(-100, 100), min_size=n_rows, max_size=n_rows)
        )
        for i in range(n_cols)
    }
    return pl.DataFrame(data)


@st.composite
def dataframes_with_n(draw):
    """DataFrame paired with a valid n (without replacement)."""
    df = draw(dataframes(min_rows=1, max_rows=20))
    n = draw(st.integers(min_value=0, max_value=len(df)))
    return df, n


@st.composite
def dataframes_with_n_replacement(draw):
    """DataFrame paired with a valid n (with replacement, n may exceed rows)."""
    df = draw(dataframes(min_rows=1, max_rows=20))
    n = draw(st.integers(min_value=0, max_value=len(df) * 3))
    return df, n


@st.composite
def dataframes_with_fraction(draw):
    """DataFrame paired with a valid fraction in [0.0, 1.0]."""
    df = draw(dataframes(min_rows=1, max_rows=20))
    fraction = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    return df, fraction


# ── Unit tests (deterministic) ────────────────────────────────────────────────


class TestReturnType:
    def test_returns_lazyframe(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        result = lazy_sample(df.lazy(), n=2, seed=0)
        assert isinstance(result, pl.LazyFrame)

    def test_collect_returns_dataframe(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        result = lazy_sample(df.lazy(), n=2, seed=0).collect()
        assert isinstance(result, pl.DataFrame)


class TestSchema:
    def test_schema_preserved_with_n(self):
        df = pl.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
        result = lazy_sample(df.lazy(), n=2, seed=0).collect()
        assert result.schema == df.schema

    def test_schema_preserved_with_fraction(self):
        df = pl.DataFrame({"x": [1, 2, 3, 4], "y": ["a", "b", "c", "d"]})
        result = lazy_sample(df.lazy(), fraction=0.5, seed=0).collect()
        assert result.schema == df.schema

    def test_no_extra_columns(self):
        df = pl.DataFrame({"a": range(10)})
        result = lazy_sample(df.lazy(), n=5, seed=0).collect()
        assert result.columns == df.columns


class TestRowCount:
    def test_n_exact_row_count(self):
        df = pl.DataFrame({"a": range(10)})
        for n in [0, 1, 5, 10]:
            result = lazy_sample(df.lazy(), n=n, seed=0).collect()
            assert len(result) == n, f"Expected {n} rows, got {len(result)}"

    def test_fraction_row_count_matches_native(self):
        df = pl.DataFrame({"a": range(20)})
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            lazy_result = lazy_sample(df.lazy(), fraction=frac, seed=0).collect()
            native_result = df.sample(fraction=frac, seed=0)
            assert len(lazy_result) == len(native_result), (
                f"fraction={frac}: lazy={len(lazy_result)}, native={len(native_result)}"
            )

    def test_with_replacement_allows_n_greater_than_rows(self):
        df = pl.DataFrame({"a": range(5)})
        result = lazy_sample(df.lazy(), n=20, with_replacement=True, seed=0).collect()
        assert len(result) == 20

    def test_empty_sample_n_zero(self):
        df = pl.DataFrame({"a": range(10)})
        result = lazy_sample(df.lazy(), n=0, seed=0).collect()
        assert len(result) == 0

    def test_fraction_zero_returns_empty(self):
        df = pl.DataFrame({"a": range(10)})
        result = lazy_sample(df.lazy(), fraction=0.0, seed=0).collect()
        assert len(result) == 0

    def test_fraction_one_returns_all_rows(self):
        df = pl.DataFrame({"a": range(10)})
        result = lazy_sample(df.lazy(), fraction=1.0, seed=0).collect()
        assert len(result) == len(df)


class TestDeterminism:
    """Same seed → same result; different seed → (very likely) different result."""

    def test_same_seed_same_result_n(self):
        df = pl.DataFrame({"a": range(20)})
        r1 = lazy_sample(df.lazy(), n=10, seed=42).collect()
        r2 = lazy_sample(df.lazy(), n=10, seed=42).collect()
        assert_frame_equal(r1, r2)

    def test_same_seed_same_result_fraction(self):
        df = pl.DataFrame({"a": range(20)})
        r1 = lazy_sample(df.lazy(), fraction=0.5, seed=42).collect()
        r2 = lazy_sample(df.lazy(), fraction=0.5, seed=42).collect()
        assert_frame_equal(r1, r2)

    def test_matches_native_same_seed_n(self):
        """lazy_sample and df.sample must select the same row indices with the same seed."""
        df = pl.DataFrame({"a": range(15)})
        lazy_result = lazy_sample(df.lazy(), n=7, seed=99).collect()
        native_result = df.sample(n=7, seed=99)
        assert_frame_equal(lazy_result, native_result, check_row_order=False)

    def test_matches_native_same_seed_fraction(self):
        df = pl.DataFrame({"a": range(20)})
        lazy_result = lazy_sample(df.lazy(), fraction=0.5, seed=7).collect()
        native_result = df.sample(fraction=0.5, seed=7)
        assert_frame_equal(lazy_result, native_result, check_row_order=False)

    def test_matches_native_with_replacement(self):
        df = pl.DataFrame({"a": range(10)})
        lazy_result = lazy_sample(
            df.lazy(), n=10, with_replacement=True, seed=3
        ).collect()
        native_result = df.sample(n=10, with_replacement=True, seed=3)
        assert_frame_equal(lazy_result, native_result, check_row_order=False)


class TestWithoutReplacement:
    def test_no_duplicate_rows_without_replacement(self):
        df = pl.DataFrame({"a": range(20)})
        result = lazy_sample(df.lazy(), n=15, with_replacement=False, seed=0).collect()
        assert result["a"].n_unique() == len(result)

    def test_all_sampled_rows_exist_in_original(self):
        df = pl.DataFrame({"a": range(10), "b": list("abcdefghij")})
        result = lazy_sample(df.lazy(), n=5, seed=0).collect()
        original_set = set(zip(df["a"].to_list(), df["b"].to_list()))
        result_set = set(zip(result["a"].to_list(), result["b"].to_list()))
        assert result_set.issubset(original_set)


class TestWithReplacement:
    def test_can_have_duplicate_rows(self):
        """With a tiny frame and a large n, duplicates are essentially certain."""
        df = pl.DataFrame({"a": [1, 2]})
        result = lazy_sample(df.lazy(), n=50, with_replacement=True, seed=0).collect()
        assert len(result) == 50
        # Values must still come from the original pool
        assert set(result["a"].to_list()).issubset({1, 2})

    def test_sampled_values_subset_of_original(self):
        df = pl.DataFrame({"a": range(5)})
        result = lazy_sample(df.lazy(), n=20, with_replacement=True, seed=1).collect()
        assert set(result["a"].to_list()).issubset(set(df["a"].to_list()))


class TestShuffle:
    def test_shuffle_does_not_change_row_count(self):
        df = pl.DataFrame({"a": range(10)})
        r_shuffled = lazy_sample(df.lazy(), n=5, shuffle=True, seed=0).collect()
        r_unshuffled = lazy_sample(df.lazy(), n=5, shuffle=False, seed=0).collect()
        assert len(r_shuffled) == len(r_unshuffled)

    def test_shuffle_same_multiset_as_unshuffled(self):
        """Shuffling changes order but not the selected rows (same seed)."""
        df = pl.DataFrame({"a": range(10)})
        r_shuffled = lazy_sample(df.lazy(), n=5, shuffle=True, seed=42).collect()
        r_unshuffled = lazy_sample(df.lazy(), n=5, shuffle=False, seed=42).collect()
        assert_frame_equal(r_shuffled, r_unshuffled, check_row_order=False)

    def test_shuffle_matches_native_multiset(self):
        df = pl.DataFrame({"a": range(10)})
        lazy_result = lazy_sample(df.lazy(), n=5, shuffle=True, seed=21).collect()
        native_result = df.sample(n=5, shuffle=True, seed=21)
        assert_frame_equal(lazy_result, native_result, check_row_order=False)


class TestEdgeCases:
    def test_single_row_dataframe_n1(self):
        df = pl.DataFrame({"a": [42]})
        result = lazy_sample(df.lazy(), n=1, seed=0).collect()
        assert len(result) == 1
        assert result["a"][0] == 42

    def test_single_row_dataframe_fraction_one(self):
        df = pl.DataFrame({"a": [42]})
        result = lazy_sample(df.lazy(), fraction=1.0, seed=0).collect()
        assert len(result) == 1

    def test_sample_all_rows_no_replacement(self):
        df = pl.DataFrame({"a": range(10)})
        result = lazy_sample(df.lazy(), n=10, with_replacement=False, seed=0).collect()
        assert_frame_equal(result, df, check_row_order=False)

    def test_multiple_columns_preserved(self):
        df = pl.DataFrame(
            {
                "int_col": [1, 2, 3, 4, 5],
                "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
                "str_col": ["a", "b", "c", "d", "e"],
            }
        )
        result = lazy_sample(df.lazy(), n=3, seed=0).collect()
        assert result.columns == df.columns
        assert result.dtypes == df.dtypes

    def test_n_equals_zero_returns_empty_with_correct_schema(self):
        df = pl.DataFrame({"x": pl.Series([1, 2, 3], dtype=pl.Int32)})
        result = lazy_sample(df.lazy(), n=0, seed=0).collect()
        assert result.schema == df.schema
        assert len(result) == 0


class TestErrors:
    def test_raises_if_n_exceeds_rows_without_replacement(self):
        df = pl.DataFrame({"a": range(3)})
        with pytest.raises(Exception):
            lazy_sample(df.lazy(), n=10, with_replacement=False, seed=0).collect()


# ── Hypothesis property-based tests ──────────────────────────────────────────


@given(df_n=dataframes_with_n())
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_property_row_count_matches_n(df_n):
    df, n = df_n
    result = lazy_sample(df.lazy(), n=n, seed=0).collect()
    assert len(result) == n


@given(df_n=dataframes_with_n())
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_property_no_duplicates_without_replacement(df_n):
    df, n = df_n
    assume(n > 0)
    # Each sampled row index should be unique; verify via a unique key column
    # We add an index to the original to track identity
    indexed = df.with_row_index("__row_id")
    lazy_indexed = indexed.lazy()
    res = lazy_sample(lazy_indexed, n=n, seed=1).collect()
    assert res["__row_id"].n_unique() == n


@given(df_n=dataframes_with_n_replacement())
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_property_row_count_with_replacement(df_n):
    df, n = df_n
    result = lazy_sample(df.lazy(), n=n, with_replacement=True, seed=2).collect()
    assert len(result) == n


@given(df_frac=dataframes_with_fraction())
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_property_fraction_row_count_matches_native(df_frac):
    df, fraction = df_frac
    lazy_result = lazy_sample(df.lazy(), fraction=fraction, seed=5).collect()
    native_result = df.sample(fraction=fraction, seed=5)
    assert len(lazy_result) == len(native_result)


@given(df_n=dataframes_with_n())
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_property_schema_invariant(df_n):
    df, n = df_n
    result = lazy_sample(df.lazy(), n=n, seed=0).collect()
    assert result.schema == df.schema
    assert result.columns == df.columns


@given(df_n=dataframes_with_n())
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_property_sampled_rows_are_subset_of_original(df_n):
    """Every sampled row must actually exist in the original frame."""
    df, n = df_n
    assume(n > 0)
    # Tag each original row with a unique id so we can track it
    indexed = df.with_row_index("__row_id")
    result = lazy_sample(indexed.lazy(), n=n, seed=42).collect()
    original_ids = set(indexed["__row_id"].to_list())
    result_ids = set(result["__row_id"].to_list())
    assert result_ids.issubset(original_ids)


@given(df_n=dataframes_with_n())
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_property_determinism(df_n):
    """Two calls with the same seed return identical frames."""
    df, n = df_n
    r1 = frame.lazy_sample(df.lazy(), n=n, seed=77).collect()
    r2 = lazy_sample(df.lazy(), n=n, seed=77).collect()
    testing.assert_frame_equal(r1, r2)


@given(df_n=dataframes_with_n())
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_property_multiset_matches_native_n(df_n):
    """lazy_sample and df.sample select the same multiset of rows (same seed)."""
    df, n = df_n
    lazy_result = lazy_sample(df.lazy(), n=n, seed=13).collect()
    native_result = df.sample(n=n, seed=13)
    assert_frame_equal(lazy_result, native_result, check_row_order=False)


@given(df_frac=dataframes_with_fraction())
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_property_multiset_matches_native_fraction(df_frac):
    """lazy_sample and df.sample select the same multiset for fraction sampling."""
    df, fraction = df_frac
    lazy_result = lazy_sample(df.lazy(), fraction=fraction, seed=13).collect()
    native_result = df.sample(fraction=fraction, seed=13)
    assert_frame_equal(lazy_result, native_result, check_row_order=False)
