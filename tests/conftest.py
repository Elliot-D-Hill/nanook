import polars as pl
import pytest


@pytest.fixture
def splits() -> dict[str, float]:
    return {"a": 0.5, "b": 0.25, "c": 0.25}


@pytest.fixture
def lf() -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "split": ["a", "a", "a", "a", "a", "a", "b", "b", "c", "c"],
            "id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "time": [1, 2, 2, 3, 1, 1, 2, 3, 1, 2],
            "a": [1.0, 1.0, 1.0, 2.0, None, 3.0, 4.0, 5.0, 6.0, 7.0],
            "b": [None, 2, 3, 1, 2, 1, 3, 4, 5, 6],
            "c": [0, 0, 0, 0, 0, None, 1, 1, 1, 1],
        }
    )


# TODO usused
@pytest.fixture
def lf_splits():
    return pl.LazyFrame(
        {
            "split": ["a", "a", "b", "c", "a", "a", "b", "c"],
            "time": [1, 2, 3, 4, 1, 2, 3, 4],
            "id": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
