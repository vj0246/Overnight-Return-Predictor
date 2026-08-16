"""
statistics.csv computation, per PDF Section 3.3/3.4.

Notation matches the PDF exactly: a = actual_return_pct, m = pred_magnitude_pct,
d = pred_direction (+-1), p = conf_direction, c = conf_magnitude,
correct = 1 if sign(a) == d else 0.

This module is unit-tested (see test_evaluate.py) against the PDF's own worked
toy-example numbers in Section 3.4 before being trusted on real predictions -
the PDF hands us a checkable correctness gate and we use it as one.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _correct(d: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    correct = 1 if sign(a) == d else 0, per PDF Section 3.3. Uses the SAME
    zero-breaking convention as actual_direction (Section 3.2: "break exact
    zeros to +1") rather than literal np.sign() (which gives 0 for exact ties)
    - the PDF's own actual_direction field IS this quantity, so consistency
    with it is not a judgment call. This matters concretely: ~8% of all
    overnight returns in this dataset are exact zero (see
    data_verification_log.md), so np.sign()'s 0-for-ties would silently mark
    every zero-return row "incorrect" regardless of the model's call,
    corrupting hit_rate, brier, log_loss, and every calibration metric that
    depends on `correct` - caught via a direct calibration audit (isotonic
    fit against actual_direction matched its target exactly bucket-by-bucket,
    but the SAME check against sign(a)-based `correct` showed a spurious
    ~8pt gap - the gap was in this function, not in the calibration).
    """
    a_sign = np.where(a >= 0, 1, -1)
    return (a_sign == d).astype(float)


def direction_score(d, a):
    return np.sum(d * a) / np.sum(np.abs(a))


def directional_return_pct(d, a):
    return np.mean(d * a)


def magnitude_score(m, a):
    return 1.0 - np.sum(np.abs(m - np.abs(a))) / np.sum(np.abs(a))


def conf_direction_score(d, a, p):
    w = 2 * p - 1
    return np.sum(w * d * a) / np.sum(w * np.abs(a))


def conf_direction_lift(d, a, p):
    return conf_direction_score(d, a, p) - direction_score(d, a)


def conf_magnitude_score(c, m, a):
    err = np.abs(m - np.abs(a))
    rho, _ = spearmanr(c, -err)
    return rho


def hit_rate(d, a):
    return np.mean(_correct(d, a))


def precision_up(d, a):
    mask = d == 1
    if mask.sum() == 0:
        return np.nan, 0
    return np.mean(a[mask] >= 0), int(mask.sum())


def recall_up(d, a):
    mask = a >= 0
    if mask.sum() == 0:
        return np.nan, 0
    return np.mean(d[mask] == 1), int(mask.sum())


def f1_up(prec, rec):
    if prec is None or rec is None or np.isnan(prec) or np.isnan(rec) or (prec + rec) == 0:
        return np.nan
    return 2 * prec * rec / (prec + rec)


def brier(p, correct):
    return np.mean((p - correct) ** 2)


def brier_skill(p, correct):
    b = brier(p, correct)
    p_ref = np.mean(correct)
    b_ref = np.mean((p_ref - correct) ** 2)
    if b_ref == 0:
        return np.nan
    return 1.0 - b / b_ref


def log_loss_metric(p, correct):
    p_clipped = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(correct * np.log(p_clipped) + (1 - correct) * np.log(1 - p_clipped))


def ece_10(p, correct):
    bins = np.linspace(0, 1, 11)
    bin_idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    N = len(p)
    total = 0.0
    for b in range(10):
        mask = bin_idx == b
        n_b = mask.sum()
        if n_b == 0:
            continue
        acc_b = correct[mask].mean()
        mean_p_b = p[mask].mean()
        total += (n_b / N) * abs(acc_b - mean_p_b)
    return total


def mae(m, a):
    return np.mean(np.abs(m - np.abs(a)))


def rmse(m, a):
    return np.sqrt(np.mean((m - np.abs(a)) ** 2))


def rank_ic_by_day(df: pd.DataFrame) -> tuple[float, float, int]:
    """
    df must have columns: pred_date, pred_magnitude_pct, actual_magnitude_pct
    (i.e. m and |a|). Computes daily cross-sectional Spearman(m, |a|), then
    mean and t-stat across days. n_obs = number of days.
    """
    daily_corrs = []
    for _, g in df.groupby("pred_date"):
        if len(g) < 3:
            continue
        rho, _ = spearmanr(g["pred_magnitude_pct"], g["actual_magnitude_pct"])
        if not np.isnan(rho):
            daily_corrs.append(rho)
    daily_corrs = np.array(daily_corrs)
    n_days = len(daily_corrs)
    if n_days == 0:
        return np.nan, np.nan, 0
    mean_ic = daily_corrs.mean()
    if n_days > 1 and daily_corrs.std() > 0:
        t_stat = mean_ic / (daily_corrs.std() / np.sqrt(n_days))
    else:
        t_stat = np.nan
    return mean_ic, t_stat, n_days


def r2_vs_vol(m, a, v):
    """v = trailing 20-day mean of |a| for that stock, excluding current day."""
    valid = ~np.isnan(v)
    m_, a_, v_ = m[valid], a[valid], v[valid]
    num = np.sum((m_ - np.abs(a_)) ** 2)
    den = np.sum((v_ - np.abs(a_)) ** 2)
    if den == 0:
        return np.nan
    return 1.0 - num / den


def mae_conf_decile(c, m, a, top: bool):
    err = np.abs(m - np.abs(a))
    thresh = np.quantile(c, 0.9 if top else 0.1)
    mask = c >= thresh if top else c <= thresh
    if mask.sum() == 0:
        return np.nan, 0
    return err[mask].mean(), int(mask.sum())


def frac_stocks_hit_gt_50(df: pd.DataFrame, min_days: int = 20):
    """df: symbol, pred_direction, actual_return_pct."""
    rows = []
    for sym, g in df.groupby("symbol"):
        if len(g) < min_days:
            continue
        hr = hit_rate(g["pred_direction"].values, g["actual_return_pct"].values)
        rows.append(hr)
    rows = np.array(rows)
    n = len(rows)
    if n == 0:
        return np.nan, 0
    return np.mean(rows > 0.50), n


def frac_stocks_beat_naive(df: pd.DataFrame, min_days: int = 20):
    rows = []
    for sym, g in df.groupby("symbol"):
        if len(g) < min_days:
            continue
        hr = hit_rate(g["pred_direction"].values, g["actual_return_pct"].values)
        naive_up_rate = np.mean(g["actual_return_pct"].values >= 0)
        rows.append(hr > naive_up_rate)
    rows = np.array(rows)
    n = len(rows)
    if n == 0:
        return np.nan, 0
    return np.mean(rows), n


def var_share_universe(a, residual):
    var_a = np.var(a)
    var_resid = np.var(residual)
    if var_a == 0:
        return np.nan
    return 1.0 - var_resid / var_a


def compute_pooled_metrics(df: pd.DataFrame, split_name: str, magnitude_extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    df columns required: pred_date, symbol, pred_magnitude_pct, pred_direction,
    conf_direction, conf_magnitude, actual_return_pct, actual_magnitude_pct.
    magnitude_extra: optional df with trailing-20d-vol column 'trail_vol_20d'
    aligned to df's index, needed for r2_vs_vol.
    """
    d = df["pred_direction"].values
    a = df["actual_return_pct"].values
    m = df["pred_magnitude_pct"].values
    p = df["conf_direction"].values
    c = df["conf_magnitude"].values
    correct = _correct(d, a)
    n = len(df)

    rows = []
    def add(metric, value, n_obs):
        rows.append(dict(split=split_name, scope="pooled", metric=metric, value=value, n_obs=n_obs))

    add("direction_score", direction_score(d, a), n)
    add("directional_return_pct", directional_return_pct(d, a), n)
    add("magnitude_score", magnitude_score(m, a), n)
    add("conf_direction_score", conf_direction_score(d, a, p), n)
    add("conf_direction_lift", conf_direction_lift(d, a, p), n)
    add("conf_magnitude_score", conf_magnitude_score(c, m, a), n)
    add("hit_rate", hit_rate(d, a), n)

    prec, n_prec = precision_up(d, a)
    add("precision_up", prec, n_prec)
    rec, n_rec = recall_up(d, a)
    add("recall_up", rec, n_rec)
    add("f1_up", f1_up(prec, rec), n)

    add("brier", brier(p, correct), n)
    add("brier_skill", brier_skill(p, correct), n)
    add("log_loss", log_loss_metric(p, correct), n)
    add("ece_10", ece_10(p, correct), n)

    add("mae", mae(m, a), n)
    add("rmse", rmse(m, a), n)

    ric, ric_t, n_days = rank_ic_by_day(df[["pred_date", "pred_magnitude_pct", "actual_magnitude_pct"]])
    add("rank_ic", ric, n_days)
    add("rank_ic_t", ric_t, n_days)

    if magnitude_extra is not None and "trail_vol_20d" in magnitude_extra.columns:
        v = magnitude_extra["trail_vol_20d"].values
        add("r2_vs_vol", r2_vs_vol(m, a, v), n)
    else:
        add("r2_vs_vol", np.nan, 0)

    mae_top, n_top = mae_conf_decile(c, m, a, top=True)
    add("mae_conf_top_decile", mae_top, n_top)
    mae_bot, n_bot = mae_conf_decile(c, m, a, top=False)
    add("mae_conf_bottom_decile", mae_bot, n_bot)
    grad = (mae_bot - mae_top) if (not np.isnan(mae_top) and not np.isnan(mae_bot)) else np.nan
    add("conf_mag_gradient", grad, n)

    frac_hit, n_sym1 = frac_stocks_hit_gt_50(df[["symbol", "pred_direction", "actual_return_pct"]])
    add("frac_stocks_hit_gt_50", frac_hit, n_sym1)
    frac_beat, n_sym2 = frac_stocks_beat_naive(df[["symbol", "pred_direction", "actual_return_pct"]])
    add("frac_stocks_beat_naive", frac_beat, n_sym2)

    residual = a - df["universe_mean_pct"].values
    add("var_share_universe", var_share_universe(a, residual), n)

    return pd.DataFrame(rows)


def compute_residual_metrics(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Residual scope: same headline + calibration metrics, computed against
    actual_return_pct - universe_mean_pct instead of actual_return_pct.
    Verified against the PDF's own Section 3.4 worked example: pooled has 25
    rows, residual has 16. The 9 pooled-only metrics (mae, rmse, r2_vs_vol,
    mae_conf_top_decile, mae_conf_bottom_decile, conf_mag_gradient,
    frac_stocks_hit_gt_50, frac_stocks_beat_naive, var_share_universe) are
    magnitude/breadth/decomposition metrics the PDF states are "pooled only -
    residualising a magnitude error is not meaningful". rank_ic/rank_ic_t ARE
    in the residual set per the worked example, computed against |residual|
    rather than |a| (magnitude error itself stays pooled-only, but magnitude's
    RANK correlation vs the residual is a distinct, meaningful, listed metric).
    """
    d = df["pred_direction"].values
    a_pooled = df["actual_return_pct"].values
    a_resid = a_pooled - df["universe_mean_pct"].values
    m = df["pred_magnitude_pct"].values
    p = df["conf_direction"].values
    correct = _correct(d, a_resid)
    n = len(df)

    rows = []
    def add(metric, value, n_obs):
        rows.append(dict(split=split_name, scope="residual", metric=metric, value=value, n_obs=n_obs))

    add("direction_score", direction_score(d, a_resid), n)
    add("directional_return_pct", directional_return_pct(d, a_resid), n)
    add("magnitude_score", magnitude_score(m, a_resid), n)
    add("conf_direction_score", conf_direction_score(d, a_resid, p), n)
    add("conf_direction_lift", conf_direction_lift(d, a_resid, p), n)
    add("conf_magnitude_score", conf_magnitude_score(df["conf_magnitude"].values, m, a_resid), n)
    add("hit_rate", hit_rate(d, a_resid), n)

    prec, n_prec = precision_up(d, a_resid)
    add("precision_up", prec, n_prec)
    rec, n_rec = recall_up(d, a_resid)
    add("recall_up", rec, n_rec)
    add("f1_up", f1_up(prec, rec), n)

    add("brier", brier(p, correct), n)
    add("brier_skill", brier_skill(p, correct), n)
    add("log_loss", log_loss_metric(p, correct), n)
    add("ece_10", ece_10(p, correct), n)

    df_resid = df.copy()
    df_resid["actual_magnitude_pct_resid"] = np.abs(a_resid)
    ric, ric_t, n_days = rank_ic_by_day(
        df_resid[["pred_date", "pred_magnitude_pct"]].assign(actual_magnitude_pct=df_resid["actual_magnitude_pct_resid"])
    )
    add("rank_ic", ric, n_days)
    add("rank_ic_t", ric_t, n_days)

    return pd.DataFrame(rows)
