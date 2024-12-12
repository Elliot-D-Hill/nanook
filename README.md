# Polars data science utilities

Nanuk is the Inuit word for polar bear.

## Usage

Nanook makes creating and transforming complex data splits easy. You can assign an arbitrary number of splits using one or more grouping columns, then apply a series of transformations with the flexibility conferred by the Polars expression API.

```python
import polars as pl
from nanook import transform

frame = pl.LazyFrame(
    {
        "id": [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9],
        "time": [1, 2, 2, 3, 1, 1, 2, 3, 1, 2, 1, 1, 1, 1],
        "a": [1, 1, 1, 2, None, 3, 4, 5, 6, 7, 8, None, 10, 11],
        "b": [None, 2, 3, 1, 2, 1, 3, 4, 5, 6, 7, 8, 9, 10],
        "c": [0, 0, 0, 0, 0, None, 1, 1, 1, 1, 1, 1, 1, 1],
    }
)

splits = {"train": 0.5, "val": 0.25, "test": 0.25}
frame = transform.assign_splits(frame=frame, splits=splits, by="id", name="split")
print(frame)
```

```python
columns = pl.col("a", "b", "c")
train = columns.filter(pl.col("split").eq("train"))
imputation = transform.impute(columns, method="median", train=train)
scale = transform.standardize(columns, method="zscore", train=train)
transforms = [imputation, scale]
df = transform.pipeline(frame=df, transforms=transforms, over="time")
print(df)
```
