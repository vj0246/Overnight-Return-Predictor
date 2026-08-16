"""
Feature construction. Three families per PLAN.md Section 2:
  2a. price/return (daily bars)
  2b. minute-bar aggregated to one row per (symbol, day)
  2c. cross-sectional (computed per-day across the realized universe)

Leakage discipline enforced throughout:
  - Every rolling/trailing feature for pred_date T uses data with date <= T only.
    Feature at row T is allowed to use T's OWN close/return/volume (T's session is
    over by the time we're predicting T->T+1, so it's inside F(T)) but never T+1.
  - No global (full-sample) statistics are used for scaling here - only per-day
    cross-sectional stats (2c) or per-symbol trailing stats, both computed from
    the past/present only. Any actual scaler/imputer fit-on-train-only happens
    later, in models.py / main.py, not here.
"""
import numpy as np
import pandas as pd


# ----------------------------- 2a. price/return -----------------------------

def price_return_features(daily_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    daily_df: single symbol, columns [date, open, high, low, close, volume, symbol],
    sorted by date ascending.
    Returns a frame indexed by date with feature columns, aligned so that the row
    for date T contains only information available at/before close(T).
    """
    df = daily_df.sort_values("date").reset_index(drop=True).copy()
    close = df["close"]
    open_ = df["open"]

    ret_1d = close.pct_change() * 100.0
    df["ret_1d"] = ret_1d

    price_windows = cfg["features"]["price_windows"]
    vol_windows = cfg["features"]["vol_windows"]

    for w in price_windows:
        df[f"ret_{w}d"] = close.pct_change(w) * 100.0

    for w in vol_windows:
        df[f"rvol_{w}d"] = ret_1d.rolling(w, min_periods=max(5, w // 2)).std()

    # overnight vs intraday decomposition, historical (uses only past-day splits)
    overnight_ret = (open_ / close.shift(1) - 1.0) * 100.0   # close_{t-1} -> open_t
    intraday_ret = (close / open_ - 1.0) * 100.0             # open_t -> close_t
    df["overnight_ret_hist"] = overnight_ret
    df["intraday_ret_hist"] = intraday_ret

    for w in [10, 20, 60]:
        df[f"overnight_mean_{w}d"] = overnight_ret.rolling(w, min_periods=max(5, w // 2)).mean()
        df[f"overnight_std_{w}d"] = overnight_ret.rolling(w, min_periods=max(5, w // 2)).std()
        df[f"intraday_mean_{w}d"] = intraday_ret.rolling(w, min_periods=max(5, w // 2)).mean()
        df[f"intraday_std_{w}d"] = intraday_ret.rolling(w, min_periods=max(5, w // 2)).std()

    # TugOfWar feature, per Lou, Polk & Skouras (2019, JFE) "A Tug of War: Overnight
    # versus Intraday Expected Returns" - the smoothed spread between a stock's own
    # overnight and intraday return components. The paper's central finding: overnight
    # and intraday returns are driven by different investor clienteles and often carry
    # OPPOSITE signal for a stock's future return - a persistently positive spread
    # (stock consistently gains overnight, gives back intraday, or vice versa) is
    # itself economically informative, not noise. Computed as EWMA(overnight_ret) -
    # EWMA(intraday_ret), halflife=60 trading days matching the paper's own smoothing
    # window. Both EWMAs use only overnight_ret/intraday_ret values through the
    # CURRENT row (pandas .ewm() is inherently causal - each row's output only uses
    # that row and earlier ones - so this is F(T)-safe by construction, same as the
    # existing rolling-window features above).
    overnight_ewma = overnight_ret.ewm(halflife=60, min_periods=20).mean()
    intraday_ewma = intraday_ret.ewm(halflife=60, min_periods=20).mean()
    df["tug_of_war"] = overnight_ewma - intraday_ewma

    # gap-fade / gap-momentum: rolling correlation of a day's own overnight gap
    # vs its own same-day intraday follow-through
    for w in [20, 60]:
        df[f"gap_fade_corr_{w}d"] = overnight_ret.rolling(w, min_periods=max(10, w // 2)).corr(intraday_ret)

    # distance from rolling high/low
    for w in [20, 60]:
        roll_high = close.rolling(w, min_periods=max(5, w // 2)).max()
        roll_low = close.rolling(w, min_periods=max(5, w // 2)).min()
        df[f"dist_high_{w}d"] = (close / roll_high - 1.0) * 100.0
        df[f"dist_low_{w}d"] = (close / roll_low - 1.0) * 100.0

    # trailing flat-open rate: fraction of the last N sessions with an exact
    # zero overnight print for THIS stock - directly informed by the 8.1%
    # exact-zero finding (see data_verification_log.md)
    is_flat = (overnight_ret == 0).astype(float)
    for w in [60, 120]:
        df[f"flat_open_rate_{w}d"] = is_flat.rolling(w, min_periods=max(10, w // 2)).mean()

    # days-since-last-extreme-gap: recognizes post-halt-style reopen regimes
    # (see FORCEMOT +31% reopen finding). "extreme" = |overnight_ret| > 10%,
    # a threshold chosen to flag halt-reopen-scale moves, not ordinary volatility.
    extreme_gap = overnight_ret.abs() > 10.0
    last_extreme_idx = pd.Series(np.where(extreme_gap, np.arange(len(df)), np.nan))
    last_extreme_idx = last_extreme_idx.ffill()
    days_since_extreme = np.arange(len(df)) - last_extreme_idx
    df["days_since_extreme_gap"] = days_since_extreme.fillna(999).clip(upper=999)

    # calendar gap feature (T+1 may be several calendar days later) - this is
    # about the NEXT session, computed from T's own date only, so it is NOT
    # leakage: it just encodes "is a weekend/holiday coming", knowable at T.
    # We don't know the exact next trading date at T without peeking, so we
    # approximate with day-of-week (Friday -> long gap likely) plus a
    # generic "is this a pre-holiday-cluster day" signal is out of scope for
    # now (would need a holiday calendar - not supplied, flagged as dropped).
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    # simple momentum/mean-reversion positioning
    for w in [5, 20]:
        df[f"zscore_close_{w}d"] = (
            (close - close.rolling(w, min_periods=max(5, w // 2)).mean())
            / close.rolling(w, min_periods=max(5, w // 2)).std()
        )

    # trailing 20-day mean of |actual_return_pct|, EXCLUDING the current day -
    # this is v in the r2_vs_vol metric formula (PDF Section 3.3), not a model
    # feature. shift(1) before the rolling window guarantees "excluding the
    # current day" as specified.
    df["trail_vol_20d"] = ret_1d.abs().shift(1).rolling(20, min_periods=10).mean()

    feature_cols = [c for c in df.columns if c not in
                    ["date", "open", "high", "low", "close", "volume", "symbol"]]
    result = df[["date", "symbol"] + feature_cols].copy()
    return result


# ----------------------------- 2b. minute-aggregated -----------------------------

def minute_daily_aggregates(minute_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates raw 1-min bars into ONE row per session per symbol. Every stat
    here uses only that session's own bars (<=15:29 of day T), so it's safely
    inside F(T) with no shift needed - the shift/rolling happens afterward
    in minute_rolling_features.
    """
    df = minute_df.copy()
    df["session_date"] = df["timestamp"].dt.normalize()

    def session_stats(g: pd.DataFrame) -> pd.Series:
        px = g["close"].values
        vol = g["volume"].values
        n = len(g)
        # 1-min log returns within the session (need >=2 bars)
        if n >= 2:
            log_ret = np.diff(np.log(px))
            rv = np.sqrt(np.sum(log_ret ** 2))  # realized vol, in log-return units
            if n >= 4 and log_ret.std() > 0:
                skew = pd.Series(log_ret).skew()
                kurt = pd.Series(log_ret).kurt()
            else:
                skew, kurt = np.nan, np.nan
        else:
            rv, skew, kurt = np.nan, np.nan, np.nan

        total_vol = vol.sum()
        first30_vol = g[g["timestamp"].dt.time <= pd.Timestamp("09:45").time()]["volume"].sum()
        last30_vol = g[g["timestamp"].dt.time >= pd.Timestamp("14:59").time()]["volume"].sum()
        vol_share_first30 = first30_vol / total_vol if total_vol > 0 else np.nan
        vol_share_last30 = last30_vol / total_vol if total_vol > 0 else np.nan

        # volume concentration (Herfindahl-style) across the session's bars
        if total_vol > 0:
            shares = vol / total_vol
            vol_hhi = np.sum(shares ** 2)
        else:
            vol_hhi = np.nan

        # opening-window vol vs rest-of-session vol
        open_mask = g["timestamp"].dt.time <= pd.Timestamp("09:45").time()
        open_px = g.loc[open_mask, "close"].values
        rest_px = g.loc[~open_mask, "close"].values
        open_rv = (np.sqrt(np.sum(np.diff(np.log(open_px)) ** 2))
                   if len(open_px) >= 2 else np.nan)
        rest_rv = (np.sqrt(np.sum(np.diff(np.log(rest_px)) ** 2))
                   if len(rest_px) >= 2 else np.nan)
        open_vol_ratio = open_rv / rest_rv if (rest_rv is not None and rest_rv not in (0, np.nan) and not np.isnan(rest_rv) and rest_rv > 0) else np.nan

        # last-30-minute realized volatility, per Zhang, Zhang, Cucuringu & Qian
        # (2023, arXiv:2202.08962) "Volatility Forecasting with Machine Learning
        # and Intraday Commonality" - their single strongest empirical finding
        # across every model family tested (HAR, LASSO, XGBoost, MLP, LSTM) was
        # that realized volatility in the LAST 30 minutes before close is the most
        # important predictor of next-day volatility, more important than any
        # other lagged feature - notable since it runs against the textbook
        # diurnal U-shape intuition that morning volatility should dominate.
        # Distinct from BOTH existing minute features: min_vol_share_last30 is a
        # VOLUME share (how much trading happened late), not a volatility measure;
        # min_open_vol_ratio compares morning-vs-rest, not closing-window level.
        # This isolates the specific closing-window RV level their paper found
        # dominant. Same log-return-sum-of-squares formula as min_realized_vol,
        # windowed to only the session's final 30 minutes (>= 15:00).
        close_mask = g["timestamp"].dt.time >= pd.Timestamp("15:00").time()
        close_px = g.loc[close_mask, "close"].values
        last30_rv = (np.sqrt(np.sum(np.diff(np.log(close_px)) ** 2))
                     if len(close_px) >= 2 else np.nan)

        # VWAP deviation: session close vs session VWAP
        vwap = (px * vol).sum() / total_vol if total_vol > 0 else np.nan
        close_vs_vwap = (px[-1] / vwap - 1.0) * 100.0 if (vwap is not None and not np.isnan(vwap) and vwap > 0) else np.nan

        # illiquidity proxy: fraction of the 375-bar session missing
        n_missing = max(0, 375 - n)
        illiq_rate = n_missing / 375.0

        return pd.Series({
            "min_realized_vol": rv,
            "min_skew": skew,
            "min_kurt": kurt,
            "min_vol_share_first30": vol_share_first30,
            "min_vol_share_last30": vol_share_last30,
            "min_vol_hhi": vol_hhi,
            "min_open_vol_ratio": open_vol_ratio,
            "min_last30_rv": last30_rv,
            "min_close_vs_vwap": close_vs_vwap,
            "min_illiq_rate": illiq_rate,
            "min_n_bars": n,
        })

    agg = df.groupby("session_date", group_keys=True).apply(session_stats, include_groups=False)
    agg = agg.reset_index().rename(columns={"session_date": "date"})
    return agg


def minute_rolling_features(agg: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Takes the one-row-per-session minute aggregates and adds trailing rolling
    means/stds, since a single day's microstructure reading is noisy - per
    PLAN.md 2b, "the informative signal is usually the recent trend/level."
    Same-day raw values (min_*) are ALSO kept as features: they are legitimately
    part of F(T) since T's own session is fully observed by close(T).
    """
    df = agg.sort_values("date").reset_index(drop=True).copy()
    windows = cfg["features"]["minute_roll_windows"]
    raw_cols = [c for c in df.columns if c.startswith("min_") and c != "min_n_bars"]

    for col in raw_cols:
        for w in windows:
            df[f"{col}_roll{w}"] = df[col].rolling(w, min_periods=max(3, w // 2)).mean()

    return df


# ----------------------------- 2c. cross-sectional -----------------------------

def cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    panel: long frame with columns [date, symbol, <feature columns...>], already
    containing per-symbol features for every (date, symbol) that is part of the
    realized universe on that date. Computes per-day rank/z-score of selected
    features, plus breadth/dispersion proxies - ALL computed only from that
    day's own cross-section, never from pooled full-sample statistics, so this
    is leakage-safe by construction (a per-day transform, not a fit-once scaler).
    """
    df = panel.copy()
    cs_source_cols = ["ret_1d", "rvol_20d", "overnight_std_20d", "overnight_mean_20d", "intraday_std_20d"]
    cs_source_cols = [c for c in cs_source_cols if c in df.columns]

    for col in cs_source_cols:
        grp = df.groupby("date")[col]
        df[f"{col}_xs_rank"] = grp.rank(pct=True)
        df[f"{col}_xs_z"] = grp.transform(lambda s: (s - s.mean()) / s.std() if s.std() > 0 else 0.0)

    # breadth / dispersion proxies (regime indicators, same value for all
    # symbols on a given date - legitimate, since "the whole universe as of
    # close(T)" is inside F(T))
    breadth = df.groupby("date")["ret_1d"].agg(
        breadth_pos_frac=lambda s: (s > 0).mean(),
        xs_dispersion=lambda s: s.std(),
    ).reset_index()
    df = df.merge(breadth, on="date", how="left")

    return df
