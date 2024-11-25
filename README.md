# Polars data science utilities

Nanuk is the Inuit word for polar bear.

## Usage

```python
import polars as pl
from nanuk import transform

df = pl.LazyFrame(
    {
        "id": [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9],
        "time": [1, 2, 2, 3, 1, 1, 2, 3, 1, 2, 1, 1, 1, 1],
        "a": [1, 1, 1, 2, None, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "b": [None, 2, 3, 1, 2, 1, 3, 4, 5, 6, 7, 8, 9, 10],
        "c": [0, 0, 0, 0, 0, None, 1, 1, 1, 1, 1, 1, 1, 1],
    }
)

splits = {"train": 0.5, "val": 0.25, "test": 0.25}
assignment = transform.assign_splits(splits=splits, group="id", name="split")
df = df.select(assignment, pl.all())

columns = pl.col("a", "b", "c")
train = columns.filter(pl.col("split").eq("train"))
imputation = transform.impute(columns, method="median", train=train)
scale = transform.standardize(columns, method="zscore", train=train)
transforms = [imputation, scale]
df = transform.pipeline(df, transforms, over="time")
df.collect()
```
