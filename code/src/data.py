"""
Data loading and target construction.

F(T) discipline: this module only ever joins close(T) to open(T+1) where T+1 is
that SYMBOL's own next trading session (not the next date on the global calendar).
This matters for symbols with idiosyncratic halts (e.g. FORCEMOT's 112-day gap) -
using the global calendar's next date would silently create a multi-month-ahead
target instead of a genuine next-session target.
"""
import json
import os
import glob
import pandas as pd
import numpy as np


def load_universe(universe_file: str) -> list[str]:
    with open(universe_file) as f:
        return json.load(f)


def resolve_paths(cfg: dict, config_path: str) -> dict:
    """
    Rewrite cfg['paths'] entries to be absolute, resolved relative to the
    config file's own directory - so the pipeline runs identically regardless
    of the caller's current working directory (no hardcoded paths, per the
    PDF's reproducibility requirement).
    """
    base = os.path.dirname(os.path.abspath(config_path))
    resolved = dict(cfg)
    resolved["paths"] = {
        k: os.path.normpath(os.path.join(base, v))
        for k, v in cfg["paths"].items()
    }
    return resolved



def load_daily(daily_dir: str, symbol: str) -> pd.DataFrame:
    path = os.path.join(daily_dir, f"{symbol}.parquet")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["symbol"] = symbol
    return df


def load_minute(minute_dir: str, symbol: str) -> pd.DataFrame:
    path = os.path.join(minute_dir, f"{symbol}.parquet")
    df = pd.read_parquet(path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["symbol"] = symbol
    return df


def build_targets_for_symbol(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (pred_date=T, target_date=T+1) for a single symbol, using that
    symbol's OWN next row as T+1 - correct even across halts/gaps, since we never
    reference the global calendar here.

    Returns columns: pred_date, target_date, symbol, close_T, open_T1,
    actual_return_pct, actual_direction, actual_magnitude_pct.
    The last row of daily_df has no T+1 and is dropped (documented, not fudged -
    see PLAN.md Section 1).
    """
    df = daily_df.reset_index(drop=True)
    out = pd.DataFrame({
        "pred_date": df["date"].iloc[:-1].values,
        "target_date": df["date"].iloc[1:].values,
        "symbol": df["symbol"].iloc[0],
        "close_T": df["close"].iloc[:-1].values,
        "open_T1": df["open"].iloc[1:].values,
    })
    out["actual_return_pct"] = (out["open_T1"] / out["close_T"] - 1.0) * 100.0
    # sign() with exact-zero broken to +1, per spec
    out["actual_direction"] = np.where(
        out["actual_return_pct"] >= 0, 1, -1
    ).astype(int)
    out["actual_magnitude_pct"] = out["actual_return_pct"].abs()
    return out


def build_all_targets(daily_dir: str, universe: list[str]) -> pd.DataFrame:
    """
    NOTE: this is a standalone convenience helper, NOT called by main.py's
    production pipeline (which builds targets per-symbol via
    build_targets_for_symbol and computes universe_mean_pct separately in
    main.py's build_panel, AFTER min_history/split filtering - see the
    comment there for why that ordering matters). The universe_mean_pct
    computed below reflects ALL symbols with a constructible target, before
    any eligibility filtering - correct for ad-hoc exploration of the raw
    target distribution, NOT a substitute for the production panel's value.
    """
    frames = []
    for sym in universe:
        daily_df = load_daily(daily_dir, sym)
        frames.append(build_targets_for_symbol(daily_df))
    targets = pd.concat(frames, ignore_index=True)

    # universe_mean_pct: cross-sectional mean of actual_return_pct across all
    # symbols SCORED on that pred_date (not a fixed 208 - respects IPO lag/halts).
    universe_mean = targets.groupby("pred_date")["actual_return_pct"].transform("mean")
    targets["universe_mean_pct"] = universe_mean
    return targets


def assign_split(pred_dates: pd.Series, cfg: dict) -> pd.Series:
    """
    Chronological split assignment with an embargo band dropped at each boundary.
    Embargo is measured in TRADING days present in the data, not calendar days,
    per "at least five trading days of embargo" in the PDF.
    """
    train_start = pd.Timestamp(cfg["splits"]["train_start"])
    train_end = pd.Timestamp(cfg["splits"]["train_end"])
    valid_start = pd.Timestamp(cfg["splits"]["valid_start"])
    valid_end = pd.Timestamp(cfg["splits"]["valid_end"])
    test_start = pd.Timestamp(cfg["splits"]["test_start"])
    test_end = pd.Timestamp(cfg["splits"]["test_end"])
    embargo_n = cfg["splits"]["embargo_days"]

    trading_days = pd.Series(sorted(pred_dates.unique()))

    def embargo_band(boundary_end, boundary_start):
        # trading days strictly between the two split windows, both sides embargoed
        before = trading_days[trading_days <= boundary_end].iloc[-embargo_n:]
        after = trading_days[trading_days >= boundary_start].iloc[:embargo_n]
        return set(before) | set(after)

    embargo_tv = embargo_band(train_end, valid_start)
    embargo_vt = embargo_band(valid_end, test_start)
    embargoed = embargo_tv | embargo_vt

    split = pd.Series(index=pred_dates.index, dtype=object)
    split[:] = None
    split[(pred_dates >= train_start) & (pred_dates <= train_end)] = "train"
    split[(pred_dates >= valid_start) & (pred_dates <= valid_end)] = "valid"
    split[(pred_dates >= test_start) & (pred_dates <= test_end)] = "test"
    split[pred_dates.isin(embargoed)] = None  # drop embargoed rows entirely
    return split


def filter_min_history(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Drops rows within each symbol's own first `min_history_days` trading
    sessions - these rows sit in the rolling-feature warmup window, where a
    large share of the feature vector is still NaN or computed on too few
    observations to be meaningful (e.g. a 60-day realized-vol feature on a
    stock's 5th day of history). Previously declared in config.yaml
    (features.min_history_days) but never wired into the pipeline - a real
    gap, not a false alarm: 12,480 of 297,165 panel rows (4.2%) fall in this
    window pre-filter, concentrated in the 26 late-IPO symbols' earliest
    sessions and in every symbol's first ~60 trading days overall. Applied
    per-symbol on pred_date rank within that symbol's own history, not on
    calendar position, so it's correct for the late-IPO symbols regardless of
    which chronological split their first 60 days happen to fall into.
    """
    min_days = cfg["features"]["min_history_days"]
    panel = panel.sort_values(["symbol", "pred_date"]).reset_index(drop=True)
    row_rank = panel.groupby("symbol").cumcount()
    return panel[row_rank >= min_days].reset_index(drop=True)
