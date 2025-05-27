import polars as pl
import pytest
from polars import testing

from nanook import frame


@pytest.fixture
def splits() -> dict[str, float]:
    return {"a": 0.5, "b": 0.25, "c": 0.25}


def test_validate_splits(splits: dict[str, float]):
    unnormalized_splits = {"a": 50.0, "b": 25.0, "c": 25.0}
    expected_warning = "Split proportions were normalized to sum to 1.0: {'a': 0.5, 'b': 0.25, 'c': 0.25}"
    with pytest.warns(UserWarning, match=expected_warning):
        result = frame.validate_splits(unnormalized_splits)
    assert result == splits


def test_to_expr():
    assert isinstance(frame.to_expr("col"), pl.Expr)
    assert isinstance(frame.to_expr(["col1", "col2"]), pl.Expr)
    assert isinstance(frame.to_expr(pl.col("col")), pl.Expr)
    with pytest.raises(ValueError):
        frame.to_expr(123)  # type: ignore[arg-type]


def test_assign_splits(lf: pl.LazyFrame, splits: dict[str, float]):
    result = frame.assign_splits(frame=lf, splits=splits, by="id")
    testing.assert_frame_equal(result, lf)


def test_join_dataframes():
    df1 = pl.DataFrame({"id": [1, 2, 3], "val1": ["a", "b", "c"]})
    df2 = pl.DataFrame({"id": [2, 3, 4], "val2": [10, 20, 30]})
    df3 = pl.DataFrame({"id": [3, 4, 5], "val3": [True, False, True]})
    expected_inner = pl.DataFrame(
        {"id": [3], "val1": ["c"], "val2": [20], "val3": [True]}
    )
    result_inner = frame.join_frames([df1, df2, df3], on="id", how="inner")
    testing.assert_frame_equal(
        result_inner, expected_inner, check_row_order=False, check_column_order=False
    )
    lf1 = df1.lazy()
    lf2 = df2.lazy()
    expected_left_data = {
        "id": [1, 2, 3],
        "val1": ["a", "b", "c"],
        "val2": [None, 10, 20],
    }
    expected_left = pl.LazyFrame(expected_left_data)
    result_left_lf = frame.join_frames([lf1, lf2], on="id", how="left")
    testing.assert_frame_equal(
        result_left_lf, expected_left, check_row_order=False, check_column_order=False
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
    df_input = pl.DataFrame({"col1": [1], "col2": ["a"]})
    assert frame.get_column_names(df_input) == ["col1", "col2"]
    lf_input = pl.LazyFrame({"col_x": [1.0], "col_y": [True]})
    assert frame.get_column_names(lf_input) == ["col_x", "col_y"]
