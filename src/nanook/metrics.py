from typing import Sequence

import polars as pl


def _cumulative_tp_fp(
    frame: pl.DataFrame | pl.LazyFrame,
    risk: str,
    event: str,
    time: str,
    horizons: Sequence[float],
) -> pl.LazyFrame:
    if isinstance(frame, pl.DataFrame):
        frame = frame.lazy()
    horizon = "horizon"
    horizon_df = pl.LazyFrame({horizon: horizons})
    frame = frame.join(horizon_df, how="cross").sort(by=[horizon, risk])
    horizon = pl.col(horizon)
    event_time = pl.col(time)
    positive = (pl.col(event).cast(bool)) & (event_time < horizon)
    negative = event_time > horizon
    total_pos = positive.sum().over(horizon)
    total_neg = negative.sum().over(horizon)
    tp_cumulative = total_pos - positive.cum_sum().over(horizon)
    fp_cumulative = total_neg - negative.cum_sum().over(horizon)
    frame = frame.with_columns(
        tp_cumulative=tp_cumulative,
        fp_cumulative=fp_cumulative,
        total_pos=total_pos,
        total_neg=total_neg,
    )
    threshold = pl.col(risk).alias("threshold")
    cols = [
        horizon,
        threshold,
        "tp_cumulative",
        "fp_cumulative",
        "total_pos",
        "total_neg",
    ]
    return frame.select(cols)


def cumulative_roc[T: (pl.DataFrame, pl.LazyFrame)](
    frame: T, risk: str, event: str, time: str, horizons: Sequence[float]
) -> T:
    """Return long ROC(t) table with columns [horizon, threshold, fpr, tpr]."""
    cumulative = _cumulative_tp_fp(
        frame, risk=risk, event=event, time=time, horizons=horizons
    )
    true_positive_rate = (pl.col("tp_cumulative") / pl.col("total_pos")).alias("tpr")
    false_positive_rate = (pl.col("fp_cumulative") / pl.col("total_neg")).alias("fpr")
    result = (
        cumulative.with_columns([true_positive_rate, false_positive_rate])
        .select("horizon", "threshold", "fpr", "tpr")
        .sort(by=["horizon", "threshold"])
    )
    if isinstance(frame, pl.DataFrame):
        return result.collect()
    return result


def cumulative_pr[T: (pl.LazyFrame, pl.DataFrame)](
    frame: T, risk: str, event: str, time: str, horizons: Sequence[float]
) -> T:
    """Return long PR(t) table with columns [horizon, threshold, recall, precision, prevalence]."""
    horizon = pl.col("horizon")
    tp_cumulative = pl.col("tp_cumulative")
    fp_cumulative = pl.col("fp_cumulative")
    total_pos = pl.col("total_pos")
    recall = (tp_cumulative / total_pos).alias("recall")
    is_prediction = tp_cumulative + fp_cumulative > 0
    precision = tp_cumulative / (tp_cumulative + fp_cumulative)
    precision = pl.when(is_prediction).then(precision).otherwise(1.0).alias("precision")
    prevalence = (total_pos / (total_pos + pl.col("total_neg"))).alias("prevalence")
    cumulative = _cumulative_tp_fp(
        frame, risk=risk, event=event, time=time, horizons=horizons
    )
    prevalence = cumulative.unique("horizon").select(horizon, prevalence)
    result = (
        cumulative.with_columns([recall, precision])
        .join(prevalence, on=horizon, how="left")
        .select(horizon, "threshold", "recall", "precision", "prevalence")
        .sort(by=[horizon, "threshold"])
    )
    if isinstance(frame, pl.DataFrame):
        return result.collect()
    return result
