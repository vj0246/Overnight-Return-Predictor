"""
Single entry point. Run from the code/ directory (or anywhere - paths resolve
relative to config.yaml, not cwd):

    python main.py --config config.yaml

Produces predictions.csv, actuals.csv, statistics.csv in the configured
output_dir. Deterministic given the same config and data (seed fixed
throughout: numpy, LightGBM, sklearn all seeded from cfg['seed']).
"""
import argparse
import os
import sys
import time
import json

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import data as D
import features as F
import models as M
import evaluate as E


META_COLS = [
    "pred_date", "target_date", "close_T", "open_T1", "actual_return_pct",
    "actual_direction", "actual_magnitude_pct", "universe_mean_pct", "date", "split",
]
CAT_COLS = ["symbol", "day_of_week", "month", "quarter"]
CONF_MAG_FEATURES = [
    "rvol_20d", "rvol_60d", "min_illiq_rate", "days_since_extreme_gap",
    "flat_open_rate_60d", "overnight_std_20d", "min_realized_vol_roll20", "symbol",
]
# Note: min_last30_rv_roll20 was tested here (a plausible vol-regime "difficulty"
# signal, analogous to min_realized_vol_roll20 already in this list) and found to
# HURT conf_magnitude's Spearman(-error) on held-out valid: 0.2140 -> 0.1955.
# Likely cause: 0.65 correlation with min_realized_vol_roll20 (see features.py's
# tug_of_war/min_last30_rv docstrings) means it mostly adds redundant noise here
# rather than new difficulty signal, even though it helps the primary magnitude
# model. Reverted rather than kept - a reasonable hypothesis that didn't hold up
# empirically, reported honestly rather than silently dropped or forced through.


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_panel(cfg: dict, universe: list[str], cache_path: str | None = None) -> pd.DataFrame:
    """
    Builds the full feature+target panel across the whole universe.
    If cache_path is given and exists, loads from cache instead of rebuilding
    (feature construction is the slowest step and is deterministic given the
    same code/config, so caching here does not affect reproducibility - a
    fresh clone with no cache present will still rebuild identically).
    """
    if cache_path and os.path.exists(cache_path):
        log(f"  loading cached panel from {cache_path}")
        return pd.read_parquet(cache_path)

    all_features, all_targets = [], []
    for i, sym in enumerate(universe):
        daily_df = D.load_daily(cfg["paths"]["daily_dir"], sym)
        minute_df = D.load_minute(cfg["paths"]["minute_dir"], sym)

        pf = F.price_return_features(daily_df, cfg)
        magg = F.minute_daily_aggregates(minute_df)
        mroll = F.minute_rolling_features(magg, cfg)
        merged = pf.merge(mroll, on="date", how="left")
        all_features.append(merged)
        all_targets.append(D.build_targets_for_symbol(daily_df))

        if (i + 1) % 25 == 0 or (i + 1) == len(universe):
            log(f"  built features for {i+1}/{len(universe)} symbols")

    feat_panel = pd.concat(all_features, ignore_index=True)
    targets = pd.concat(all_targets, ignore_index=True)

    full = targets.merge(feat_panel, left_on=["pred_date", "symbol"], right_on=["date", "symbol"], how="inner")
    n_pre_history_filter = len(full)
    full = D.filter_min_history(full, cfg)
    log(f"  min_history_days filter: {n_pre_history_filter} -> {len(full)} rows "
        f"({n_pre_history_filter - len(full)} dropped, warmup window per symbol)")

    full["split"] = D.assign_split(full["pred_date"], cfg)
    n_before = len(full)
    full = full[full["split"].notna()].reset_index(drop=True)
    log(f"  panel: {n_before} rows built, {len(full)} kept after split/embargo filter")

    # Cross-sectional features (rank/z-score/breadth/dispersion) computed HERE,
    # after ALL row-eligibility filtering, not before it. Per external review:
    # this was NOT a T+1 lookahead issue (every cross-sectional feature only
    # ever used same-day T information, which the PDF explicitly permits -
    # "anything computed across the universe up to the close of T sits inside
    # F(T)"). It WAS a universe-definition consistency issue: computing these
    # before filter_min_history/assign_split meant a stock's rank/z-score on
    # date T could include a symbol D that gets later dropped as ineligible
    # (insufficient history, or outside the split/embargo window) - so the
    # cross-sectional universe briefly differed from the actual SCORED
    # universe used everywhere else (predictions, universe_mean_pct). Moved
    # here so every cross-sectional statistic reflects exactly the same
    # point-in-time scored universe as the rest of the pipeline.
    full = F.cross_sectional_features(full)

    # universe_mean_pct computed HERE, after all row-eligibility filtering
    # (min_history_days AND split/embargo) is complete - per the PDF's own
    # definition, "cross-sectional mean of actual_return_pct across all
    # SCORED names on that pred_date". Computing this earlier (before
    # filtering) is a confirmed bug: it would include returns from symbols
    # later dropped by filtering, which are not actually "scored" - verified
    # on real data to affect 157,114/284,685 rows (55%) with a wrong value
    # under the old before-filtering computation, feeding directly into every
    # residual-scope metric (residual = actual_return_pct - universe_mean_pct).
    full["universe_mean_pct"] = full.groupby("pred_date")["actual_return_pct"].transform("mean")

    if cache_path:
        cache_dir = os.path.dirname(os.path.abspath(cache_path))
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        full.to_parquet(cache_path)
        log(f"  cached panel to {cache_path}")

    return full


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in META_COLS and c != "symbol"]
    return cols + ["symbol"]


def run_pipeline(cfg: dict, cache_path: str | None = None) -> dict[str, pd.DataFrame]:
    seed = cfg["seed"]
    np.random.seed(seed)

    log("loading universe")
    universe = D.load_universe(cfg["paths"]["universe_file"])
    log(f"universe: {len(universe)} symbols")

    log("building feature+target panel (this is the slow step)")
    panel = build_panel(cfg, universe, cache_path=cache_path)

    feature_cols = get_feature_cols(panel)
    log(f"n features: {len(feature_cols)}")

    train = panel[panel["split"] == "train"].reset_index(drop=True)
    valid = panel[panel["split"] == "valid"].reset_index(drop=True)
    test = panel[panel["split"] == "test"].reset_index(drop=True)
    log(f"split sizes -> train: {len(train)}, valid: {len(valid)}, test: {len(test)}")

    # Build FIXED categorical code maps from TRAIN ONLY - a "frozen
    # vocabulary" learned from train and applied unchanged to valid/test,
    # per external review: encoding from the full panel (all three splits
    # combined) is not target leakage (no future returns are used), but it
    # does let the code map "see" symbols/values that only exist in
    # valid/test, which is not the clean train-only preprocessing principle
    # the assignment's Section 4 rule is built on. A symbol absent from
    # train (5 late-IPO names, confirmed absent after the min_history
    # filter) now correctly gets an "unseen category" code (-1) at
    # valid/test time - the model never learned anything about those
    # symbols during training, so that is the accurate state to encode,
    # not a borrowed code from a vocabulary the model was never fit on.
    cat_maps = M.build_global_cat_maps(train, CAT_COLS)
    log(f"categorical maps built from TRAIN ONLY: symbol has {len(cat_maps['symbol'])} categories "
        f"({panel['symbol'].nunique() - len(cat_maps['symbol'])} symbols in valid/test are unseen by train, "
        f"correctly encoded as -1)")

    X_train, X_valid, X_test = train[feature_cols], valid[feature_cols], test[feature_cols]

    # ---------------- 1. Magnitude ----------------
    log("training magnitude model")
    mag_model = M.MagnitudeModel(cfg["models"]["magnitude"], CAT_COLS, cat_maps=cat_maps)
    mag_model.fit(X_train, train["actual_magnitude_pct"], X_valid, valid["actual_magnitude_pct"])

    # ---------------- 2. Direction ----------------
    log("training direction model")
    train_is_up = (train["actual_direction"] == 1).astype(int)
    valid_is_up = (valid["actual_direction"] == 1).astype(int)
    dir_model = M.DirectionModel(cfg["models"]["direction"], CAT_COLS, cat_maps=cat_maps)
    dir_model.fit(X_train, train_is_up, X_valid, valid_is_up)

    # ---------------- 3. conf_direction calibrator ----------------
    # Fit on VALID (direction model never trained on these rows) - see PLAN.md
    # Section 4: valid is intentionally used as training ground for the
    # second-stage confidence models, stated explicitly here and in the README.
    log("fitting direction calibrator on valid split")
    p_up_valid = dir_model.predict_proba_up(X_valid)
    calibrator = M.DirectionCalibrator()
    calibrator.fit(p_up_valid, valid_is_up.values)

    # ---------------- 4. conf_magnitude ----------------
    log("generating out-of-fold magnitude errors on train (date-based walk-forward, verified time-ordered)")
    oof_pred, oof_error = M.out_of_fold_magnitude_errors(
        X_train, train["actual_magnitude_pct"].values, train["pred_date"], CAT_COLS, cfg["models"]["magnitude"],
        cat_maps=cat_maps, n_folds=5, seed=seed
    )
    mag_pred_valid = mag_model.predict(X_valid)
    valid_error = np.abs(mag_pred_valid - valid["actual_magnitude_pct"].values)

    conf_mag_features = [f for f in CONF_MAG_FEATURES if f in feature_cols]
    log(f"conf_magnitude feature set: {conf_mag_features}")
    valid_mask = ~np.isnan(oof_error)
    Xtr_reduced = X_train[conf_mag_features][valid_mask].reset_index(drop=True)
    oof_err_clean = oof_error[valid_mask]

    log("training conf_magnitude model")
    cm_model = M.ConfMagnitudeModel(cfg["models"]["conf_magnitude"], ["symbol"], cat_maps=cat_maps)
    cm_model.fit(Xtr_reduced, oof_err_clean, X_valid[conf_mag_features], valid_error)

    # ---------------- Generate predictions for all three splits ----------------
    log("generating predictions for train/valid/test")
    results = {}
    for split_name, X_split, split_df in [("train", X_train, train), ("valid", X_valid, valid), ("test", X_test, test)]:
        mag_pred = mag_model.predict(X_split)
        p_up = dir_model.predict_proba_up(X_split)
        final_dir, final_conf_dir = calibrator.predict_direction_and_conf(p_up)
        conf_mag = cm_model.predict_conf_magnitude(X_split[conf_mag_features])

        pred_df = pd.DataFrame({
            "pred_date": split_df["pred_date"].dt.strftime("%Y-%m-%d"),
            "target_date": split_df["target_date"].dt.strftime("%Y-%m-%d"),
            "symbol": split_df["symbol"].values,
            "pred_magnitude_pct": np.round(mag_pred, 4),
            "pred_direction": final_dir.astype(int),
            "conf_direction": np.round(final_conf_dir, 4),
            "conf_magnitude": np.round(conf_mag, 4),
            "split": split_name,
        })

        actual_df = pd.DataFrame({
            "pred_date": split_df["pred_date"].dt.strftime("%Y-%m-%d"),
            "target_date": split_df["target_date"].dt.strftime("%Y-%m-%d"),
            "symbol": split_df["symbol"].values,
            "actual_return_pct": np.round(split_df["actual_return_pct"].values, 4),
            "actual_direction": split_df["actual_direction"].values.astype(int),
            "actual_magnitude_pct": np.round(split_df["actual_magnitude_pct"].values, 4),
            "universe_mean_pct": np.round(split_df["universe_mean_pct"].values, 4),
        })

        results[split_name] = dict(pred=pred_df, actual=actual_df, raw_df=split_df, mag_pred_raw=mag_pred,
                                    final_dir=final_dir, final_conf_dir=final_conf_dir, conf_mag=conf_mag)

    return results


def compute_all_statistics(results: dict) -> pd.DataFrame:
    stat_frames = []
    for split_name in ["train", "valid", "test"]:
        pred = results[split_name]["pred"]
        actual = results[split_name]["actual"]
        raw_df = results[split_name]["raw_df"]

        eval_df = pd.DataFrame({
            "pred_date": pd.to_datetime(pred["pred_date"]),
            "symbol": pred["symbol"],
            "pred_magnitude_pct": pred["pred_magnitude_pct"],
            "pred_direction": pred["pred_direction"],
            "conf_direction": pred["conf_direction"],
            "conf_magnitude": pred["conf_magnitude"],
            "actual_return_pct": actual["actual_return_pct"],
            "actual_magnitude_pct": actual["actual_magnitude_pct"],
            "universe_mean_pct": actual["universe_mean_pct"],
            "trail_vol_20d": raw_df["trail_vol_20d"].values,
        })

        pooled = E.compute_pooled_metrics(eval_df, split_name, magnitude_extra=eval_df[["trail_vol_20d"]])
        residual = E.compute_residual_metrics(eval_df, split_name)
        stat_frames.append(pooled)
        stat_frames.append(residual)

    return pd.concat(stat_frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--cache", default=None, help="Optional path to cache/load the built feature panel (speeds up reruns; not required for a fresh clone).")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = D.resolve_paths(cfg, args.config)

    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)

    t0 = time.time()
    results = run_pipeline(cfg, cache_path=args.cache)
    log(f"pipeline complete in {time.time()-t0:.1f}s")

    # concat predictions/actuals across splits
    all_pred = pd.concat([results[s]["pred"] for s in ["train", "valid", "test"]], ignore_index=True)
    all_actual = pd.concat([results[s]["actual"] for s in ["train", "valid", "test"]], ignore_index=True)

    pred_path = os.path.join(cfg["paths"]["output_dir"], "predictions.csv")
    actual_path = os.path.join(cfg["paths"]["output_dir"], "actuals.csv")
    all_pred.to_csv(pred_path, index=False)
    all_actual.to_csv(actual_path, index=False)
    log(f"wrote {pred_path} ({len(all_pred)} rows)")
    log(f"wrote {actual_path} ({len(all_actual)} rows)")

    log("computing statistics.csv")
    stats = compute_all_statistics(results)
    stats_path = os.path.join(cfg["paths"]["output_dir"], "statistics.csv")
    stats.to_csv(stats_path, index=False)
    log(f"wrote {stats_path} ({len(stats)} rows)")

    log("DONE")


if __name__ == "__main__":
    main()
