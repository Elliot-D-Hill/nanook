import polars as pl
import pytest
from polars import testing

from nanook import frame


@pytest.fixture
def splits() -> dict[str, float]:
    return {"a": 0.5, "b": 0.25, "c": 0.25}


def test_validate_splits(lf: pl.LazyFrame, splits: dict[str, float]):
    unnormalized_splits = {"a": 50.0, "b": 25.0, "c": 25.0}
    result = frame.validate_splits(unnormalized_splits)
    assert result == splits
    expected_warning = "Split proportions were normalized to sum to 1.0: {'a': 0.5, 'b': 0.25, 'c': 0.25}"
    with pytest.warns(UserWarning, match=expected_warning):
        frame.validate_splits(splits=unnormalized_splits)


def test_assign_splits(lf: pl.LazyFrame, splits: dict[str, float]):
    result = frame.assign_splits(frame=lf, splits=splits, by="id")
    testing.assert_frame_equal(result, lf)
