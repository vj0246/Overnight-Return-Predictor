"""
Four genuinely separate models, per PLAN.md Section 3:
  1. Magnitude regressor      - E[|r| | F(T)]
  2. Direction classifier     - sign(E[r | F(T)])
  3. conf_direction calibrator - P(emitted direction realized | F(T)), fit via
     isotonic regression on the direction classifier's OOF/valid scores -
     a genuinely distinct fitted quantity, not a rescale.
  4. conf_magnitude regressor - predicts the magnitude model's OWN expected
     error, trained on OUT-OF-FOLD magnitude residuals so it never sees
     in-sample error (which would be optimistic and answer a different,
     easier question than the one it needs to answer at inference time).

All models are pooled across the universe with `symbol` as a native LightGBM
categorical feature (not 208 separate models) - see PLAN.md Section 3/6 for
the variance-cost justification.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold


def _lgbm_params(cfg_block: dict) -> dict:
    """Extract LightGBM constructor kwargs from a config sub-block."""
    p = dict(cfg_block)
    p.pop("early_stopping_rounds", None)
    p.pop("log_target", None)
    return p


def prep_categorical(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    return X


def build_global_cat_maps(X: pd.DataFrame, cat_cols: list[str]) -> dict[str, dict]:
    """
    Builds a FIXED category->code mapping per categorical column, from the
    union of values seen across the full dataset this is called on (intended
    to be called ONCE on the full panel, before any train/valid/test split).
    This must be built once and reused for every downstream to_numpy call -
    NEVER recomputed per split. Recomputing pd.Categorical(...).codes
    independently on each split's DataFrame is a serious, confirmed bug: if
    even one category (e.g. a late-IPO symbol absent from train after the
    min_history_days filter) is missing from one split, every alphabetically-
    later category's code silently shifts in that split relative to the
    others - the model learns "code 47 means symbol X" during training and
    is then scored against "code 47 means symbol Y" at inference time.
    Verified on this project's real data: 5 of 208 symbols are absent from
    train (203 present) after the min_history filter, causing 120/203 -
    59% - of train's category codes to disagree with valid/test's codes
    under the old per-call encoding. Fixed by building one global map here
    and threading it through every model class instead.
    """
    maps = {}
    for c in cat_cols:
        categories = sorted(X[c].dropna().unique().tolist())
        maps[c] = {cat: i for i, cat in enumerate(categories)}
    return maps


def to_numpy_with_cat_codes(X: pd.DataFrame, cat_cols: list[str], cat_maps: dict[str, dict]) -> tuple[np.ndarray, list[int]]:
    """
    Converts a DataFrame to a plain, contiguous numpy float64 array, with
    categorical columns encoded via the FIXED, pre-built cat_maps (see
    build_global_cat_maps) - never recomputed per-call. A category value not
    present in cat_maps (should not happen if cat_maps was built from the
    full panel, but defensively handled) is encoded as -1, which LightGBM
    treats as a distinct, valid "unseen category" code rather than crashing
    or silently colliding with an existing code.

    Also converts to plain numpy (not pandas Categorical dtype) because of a
    real, reproduced bug: on at least one Windows Python 3.10 environment
    (numpy 1.26.4, pandas 2.2.3, lightgbm 4.7.0), passing a pandas
    DataFrame/Series directly into LightGBM's sklearn .fit()/.predict()
    crashes the native library with "OSError: exception: access violation
    reading 0x0000000000000000" inside lightgbm's set_label/set_field C call
    - reproduced consistently at real-pipeline scale but not with plain
    numpy arrays at the identical scale. Root-caused via bisection: pandas
    category dtype ruled out, scale ruled out, column count ruled out -
    isolated to "any pandas DataFrame/Series input" as the trigger. Not
    reproduced on the Linux sandbox this pipeline was originally verified
    in - an environment-specific Windows issue, not a logic bug - but
    routing through plain numpy sidesteps it entirely and is harmless
    elsewhere.
    """
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    X_out = X.copy()
    for c in cat_cols:
        X_out[c] = X_out[c].map(cat_maps[c]).fillna(-1).astype(np.float64)
    arr = np.ascontiguousarray(X_out.to_numpy(dtype=np.float64))
    return arr, cat_idx


# ----------------------------- 1. Magnitude -----------------------------

class MagnitudeModel:
    """
    Regresses log1p(|actual_return_pct|), inverse-transformed at predict time.
    log1p chosen because |r| is right-skewed (a few large post-halt/event moves,
    see FORCEMOT +31% finding) - log-space regression with L1 loss is more
    robust to those outliers than raw-space L2, and PLAN.md commits to
    comparing this against a raw-space variant on validation before locking.
    """
    def __init__(self, cfg_block: dict, cat_cols: list[str], cat_maps: dict[str, dict] = None):
        self.cfg_block = cfg_block
        self.cat_cols = cat_cols
        self.log_target = cfg_block.get("log_target", True)
        self.model = None
        self.feature_names_ = None
        self.cat_maps = cat_maps  # fixed maps, built externally from the FULL panel

    def fit(self, X_train, y_train, X_valid, y_valid):
        self.feature_names_ = list(X_train.columns)
        if self.cat_maps is None:
            # fallback: build from train ONLY if not provided - logs a warning-
            # worthy situation, since this reintroduces the train/valid code-
            # mismatch risk this whole mechanism exists to prevent. Callers
            # (main.py) should always pass cat_maps built from the full panel.
            self.cat_maps = build_global_cat_maps(X_train, self.cat_cols)
        Xtr, cat_idx = to_numpy_with_cat_codes(X_train, self.cat_cols, self.cat_maps)
        Xva, _ = to_numpy_with_cat_codes(X_valid, self.cat_cols, self.cat_maps)
        ytr = np.log1p(y_train.to_numpy(dtype=np.float64)) if self.log_target else y_train.to_numpy(dtype=np.float64)
        yva = np.log1p(y_valid.to_numpy(dtype=np.float64)) if self.log_target else y_valid.to_numpy(dtype=np.float64)

        params = _lgbm_params(self.cfg_block)
        early_stop = self.cfg_block["early_stopping_rounds"]
        self.model = lgb.LGBMRegressor(**params, random_state=42, verbosity=-1)
        self.model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            eval_metric="l1",
            feature_name=self.feature_names_,
            categorical_feature=cat_idx,
            callbacks=[lgb.early_stopping(early_stop, verbose=False)],
        )
        return self

    def predict(self, X):
        Xarr, _ = to_numpy_with_cat_codes(X[self.feature_names_], self.cat_cols, self.cat_maps)
        raw = self.model.predict(Xarr)
        pred = np.expm1(raw) if self.log_target else raw
        return np.clip(pred, 0.0, None)  # magnitude must be >= 0


# ----------------------------- 2. Direction -----------------------------

class DirectionModel:
    """Binary classifier: P(up). pred_direction derived as sign(P(up)-0.5)."""
    def __init__(self, cfg_block: dict, cat_cols: list[str], cat_maps: dict[str, dict] = None):
        self.cfg_block = cfg_block
        self.cat_cols = cat_cols
        self.model = None
        self.feature_names_ = None
        self.cat_maps = cat_maps

    def fit(self, X_train, y_train_is_up, X_valid, y_valid_is_up):
        self.feature_names_ = list(X_train.columns)
        if self.cat_maps is None:
            self.cat_maps = build_global_cat_maps(X_train, self.cat_cols)
        Xtr, cat_idx = to_numpy_with_cat_codes(X_train, self.cat_cols, self.cat_maps)
        Xva, _ = to_numpy_with_cat_codes(X_valid, self.cat_cols, self.cat_maps)
        ytr = y_train_is_up.to_numpy()
        yva = y_valid_is_up.to_numpy()

        params = _lgbm_params(self.cfg_block)
        early_stop = self.cfg_block["early_stopping_rounds"]
        self.model = lgb.LGBMClassifier(**params, random_state=42, verbosity=-1)
        self.model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            eval_metric="binary_logloss",
            feature_name=self.feature_names_,
            categorical_feature=cat_idx,
            callbacks=[lgb.early_stopping(early_stop, verbose=False)],
        )
        return self

    def predict_proba_up(self, X):
        Xarr, _ = to_numpy_with_cat_codes(X[self.feature_names_], self.cat_cols, self.cat_maps)
        return self.model.predict_proba(Xarr)[:, 1]


# ----------------------------- 3. conf_direction calibrator -----------------------------

class DirectionCalibrator:
    """
    Fits isotonic regression mapping raw P(up) -> realized P(correct | emitted
    direction), on a held-out set (valid split - the direction model never
    trained on these rows). This is a SEPARATE fitted object from the
    direction classifier: it answers "how often is a score like this one
    actually right", not "what does the classifier think".

    Isotonic chosen over Platt/sigmoid per PLAN.md Section 3: tree-ensemble
    miscalibration is often non-monotonic-sigmoidal, and isotonic makes no
    parametric shape assumption - just enforces monotonicity, which we then
    verify below is not violated on the held-out check.

    Output is capped strictly inside (0,1) via CONF_CAP - an uncapped isotonic
    fit can produce an exact 1.0 on a thin bin (few valid-split samples that
    happen to all be correct), which is an unjustified certainty claim from a
    finite sample. This was caught empirically: an uncapped fit produced
    conf_direction==1.0 on 18 test rows, 3 of which were WRONG (83% actual
    accuracy on a claimed-certain subset) - a genuine Brier-score problem, not
    a cosmetic one. Capping mirrors the spec's own log_loss clip to
    [1e-6, 1-1e-6], which implicitly acknowledges p=1 is never defensible.
    """
    CONF_CAP = 0.01  # output confined to [CONF_CAP, 1-CONF_CAP]

    def __init__(self):
        self.iso = IsotonicRegression(
            out_of_bounds="clip", y_min=self.CONF_CAP, y_max=1 - self.CONF_CAP
        )

    def fit(self, raw_p_up: np.ndarray, is_up: np.ndarray):
        # We calibrate P(up) directly against realized is_up - this lets the
        # SAME calibrator serve both an up-call and a down-call: for a down
        # call the realized "correct" probability is 1 - calibrated_P(up).
        self.iso.fit(raw_p_up, is_up)
        return self

    def calibrate_p_up(self, raw_p_up: np.ndarray) -> np.ndarray:
        return self.iso.predict(raw_p_up)

    def predict_direction_and_conf(self, raw_p_up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        THE correct entry point for production use - decides pred_direction
        from the CALIBRATED probability crossing 0.5, not the raw classifier's
        0.5 threshold. This matters: a raw score of 0.5 does not mean "50/50"
        for an uncalibrated tree ensemble, and deciding direction on the raw
        threshold while confidence is computed on the calibrated one silently
        decouples the two - if the raw model's decision boundary sits at a
        calibrated probability materially above 0.5 (which happens whenever
        the base rate is skewed, as it is here - see data_verification_log.md,
        full-universe up-rate ~70-73%), EVERY raw "down" call the flip-rule
        later inspects turns out to have calibrated P(up) > 0.5 too, and gets
        flipped back to "up" - erasing 100% of down-calls, confirmed empirically
        on the full 208-symbol run (481 raw down-calls on test, 0 survived the
        old flip-based logic). Deciding on the calibrated probability directly
        fixes this at the source: the flip rule below is then a true rare-case
        safety net (only triggers exactly at the calibrated 0.5 boundary),
        not the mechanism erasing every down-call.
        """
        p_up_cal = self.calibrate_p_up(raw_p_up)
        pred_direction = np.where(p_up_cal >= 0.5, 1, -1)
        conf = np.where(pred_direction == 1, p_up_cal, 1.0 - p_up_cal)
        # true safety net: only possible trigger is the exact boundary case
        # (p_up_cal == 0.5 exactly), kept for defensive completeness.
        flipped_direction = np.where(conf < 0.5, -pred_direction, pred_direction)
        flipped_conf = np.where(conf < 0.5, 1.0 - conf, conf)
        return flipped_direction, flipped_conf

    def conf_for_direction(self, raw_p_up: np.ndarray, pred_direction: np.ndarray) -> np.ndarray:
        """
        DEPRECATED entry point - kept only because it was the original design
        and is unit-tested against a synthetic case (see test history). Takes
        an EXTERNALLY-decided pred_direction (typically from the raw, un-
        calibrated 0.5 threshold) and only adjusts confidence/flips at the
        margin. This is what produced the always-up collapse on the full run:
        do not use for production predictions - use predict_direction_and_conf
        instead, which decides direction and confidence from the same
        (calibrated) basis so they cannot disagree about where 50/50 is.
        """
        p_up_cal = self.calibrate_p_up(raw_p_up)
        conf = np.where(pred_direction == 1, p_up_cal, 1.0 - p_up_cal)
        flipped_direction = np.where(conf < 0.5, -pred_direction, pred_direction)
        flipped_conf = np.where(conf < 0.5, 1.0 - conf, conf)
        return flipped_direction, flipped_conf


# ----------------------------- 4. conf_magnitude -----------------------------

def out_of_fold_magnitude_errors(X_train, y_train, pred_dates, cat_cols, cfg_block, cat_maps=None, n_folds=5, seed=42):
    """
    Walk-forward (expanding-window) out-of-fold generation on the TRAIN split
    only, used solely to produce out-of-fold |error| labels for training the
    conf_magnitude model. Never touches valid/test.

    CORRECTED A SECOND TIME after external review found the first "fix" was
    still broken: that version defined fold boundaries by ROW POSITION
    (np.arange over the DataFrame's existing index), which is only
    time-ordered if the DataFrame itself is sorted by date. It was not -
    filter_min_history() in data.py sorts by ["symbol", "pred_date"], and
    that order survives unchanged through main.py's split masking (boolean
    filtering + reset_index preserves relative row order, it does not
    reorder rows). Verified directly on real data: under the old code,
    "row 0:28893" (nominally "the earliest block") actually spanned symbols
    360ONE through roughly BIOCON, with a date range covering the ENTIRE
    train period (2020-08-24 to 2024-03-20) across just 33 symbols - meaning
    a fold's "past-only" training data could and did include rows from
    2024 predicting a target dated 2021, for a different symbol. That is
    exactly the K-fold-on-time-series lookahead the assignment calls an
    automatic rejection, reintroduced through indexing rather than through
    the original purged-K-fold logic.

    THIS version is genuinely date-based: fold boundaries are computed from
    the sorted UNIQUE VALUES of pred_dates, not row positions, and every
    internal split explicitly sorts by date before slicing - no assumption
    is made about the input DataFrame's existing row order. pred_dates must
    be a Series/array of the same length as X_train/y_train, aligned by
    position (not by index label) - the caller's original row order is
    preserved in the returned oof_pred/oof_error arrays regardless of what
    internal sorting this function does.

    Embargo: a 5-trading-day gap (matching the outer split's own embargo in
    data.py) is dropped between each fold's training dates and its
    validation dates, using the same trading-day-count logic as the outer
    split - not a row-count buffer, which would silently vary in calendar/
    trading-day size depending on how many symbols happen to sit near a
    fold boundary.
    """
    n = len(X_train)
    pred_dates = pd.to_datetime(pd.Series(pred_dates).reset_index(drop=True))
    X_train = X_train.reset_index(drop=True)
    y_train = pd.Series(y_train).reset_index(drop=True)

    # Sort ONLY to determine fold boundaries and to build the training/
    # validation row-index sets correctly - the returned arrays are placed
    # back into the caller's ORIGINAL (possibly symbol-major) row order.
    order = np.argsort(pred_dates.values, kind="stable")
    sorted_dates = pred_dates.values[order]

    unique_dates = np.unique(sorted_dates)
    n_dates = len(unique_dates)
    date_fold_bounds = np.linspace(0, n_dates, n_folds + 2).astype(int)
    embargo_n = 5  # trading days, matching data.py's outer-split embargo

    oof_pred = np.full(n, np.nan)

    for i in range(1, n_folds + 1):
        val_date_start = unique_dates[date_fold_bounds[i]]
        val_date_end_idx = min(date_fold_bounds[i + 1], n_dates) - 1
        val_date_end = unique_dates[val_date_end_idx]

        # embargo: drop the last `embargo_n` TRADING DATES immediately
        # before val_date_start from the training set, by date value - not
        # a row count, so it is exactly 5 trading days regardless of how
        # many symbols/rows fall on those dates.
        train_date_cutoff_idx = date_fold_bounds[i] - embargo_n
        if train_date_cutoff_idx <= 0:
            continue  # not enough prior history to embargo safely
        train_date_cutoff = unique_dates[train_date_cutoff_idx - 1]

        train_mask = pred_dates.values <= train_date_cutoff
        val_mask = (pred_dates.values >= val_date_start) & (pred_dates.values <= val_date_end)
        train_idx_orig = np.where(train_mask)[0]
        val_idx_orig = np.where(val_mask)[0]

        if len(train_idx_orig) < 200 or len(val_idx_orig) < 20:
            continue

        # sort the training rows themselves chronologically before carving
        # an early-stopping holdout off the tail - so "last 10%" genuinely
        # means the most recent dates, not an arbitrary row slice.
        train_order = np.argsort(pred_dates.values[train_idx_orig], kind="stable")
        train_idx_sorted = train_idx_orig[train_order]
        cut = int(len(train_idx_sorted) * 0.9)
        fit_idx = train_idx_sorted[:cut]
        es_idx = train_idx_sorted[cut:]

        m = MagnitudeModel(cfg_block, cat_cols, cat_maps=cat_maps)
        Xtr_fit, ytr_fit = X_train.iloc[fit_idx], y_train.iloc[fit_idx]
        Xtr_es, ytr_es = X_train.iloc[es_idx], y_train.iloc[es_idx]
        m.fit(Xtr_fit, ytr_fit, Xtr_es, ytr_es)

        Xva_fold = X_train.iloc[val_idx_orig]
        pred = m.predict(Xva_fold)
        oof_pred[val_idx_orig] = pred

    oof_error = np.abs(oof_pred - y_train.values)
    return oof_pred, oof_error


class ConfMagnitudeModel:
    """
    Second-stage regressor predicting the magnitude model's expected |error|,
    using a reduced feature set describing "how hard is this prediction"
    (recent vol regime, illiquidity, history length, the point prediction
    itself, recent error dispersion) rather than the full feature set -
    per PLAN.md Section 3, this is deliberately a different, smaller question
    than "what is the magnitude", not the same model repurposed.
    """
    def __init__(self, cfg_block: dict, cat_cols: list[str], cat_maps: dict[str, dict] = None):
        self.cfg_block = cfg_block
        self.cat_cols = cat_cols
        self.model = None
        self.feature_names_ = None
        self.train_error_dist_ = None  # for rank-transform to [0,1]
        self.cat_maps = cat_maps

    def fit(self, X_train, oof_error, X_valid, valid_error):
        self.feature_names_ = list(X_train.columns)
        if self.cat_maps is None:
            self.cat_maps = build_global_cat_maps(X_train, self.cat_cols)
        Xtr, cat_idx = to_numpy_with_cat_codes(X_train, self.cat_cols, self.cat_maps)
        Xva, _ = to_numpy_with_cat_codes(X_valid, self.cat_cols, self.cat_maps)
        oof_error = np.asarray(oof_error, dtype=np.float64)
        valid_error = np.asarray(valid_error, dtype=np.float64)

        params = _lgbm_params(self.cfg_block)
        early_stop = self.cfg_block["early_stopping_rounds"]
        self.model = lgb.LGBMRegressor(**params, random_state=42, verbosity=-1)
        self.model.fit(
            Xtr, oof_error,
            eval_set=[(Xva, valid_error)],
            eval_metric="l2",
            feature_name=self.feature_names_,
            categorical_feature=cat_idx,
            callbacks=[lgb.early_stopping(early_stop, verbose=False)],
        )
        self.train_error_dist_ = np.sort(oof_error[~np.isnan(oof_error)])
        return self

    def predict_expected_error(self, X):
        Xarr, _ = to_numpy_with_cat_codes(X[self.feature_names_], self.cat_cols, self.cat_maps)
        return np.clip(self.model.predict(Xarr), 0.0, None)

    def predict_conf_magnitude(self, X):
        """
        Transforms predicted expected error into [0,1] via a rank-transform
        against the training error distribution: conf = 1 - percentile_rank(
        predicted_error). Low predicted error -> high confidence. Chosen over
        a raw 1-normalized-error transform because rank-transform is robust
        to the error distribution's scale/outliers (again, the FORCEMOT-style
        extreme-move rows) and guarantees a well-spread [0,1] output.
        """
        pred_err = self.predict_expected_error(X)
        ranks = np.searchsorted(self.train_error_dist_, pred_err) / len(self.train_error_dist_)
        conf = 1.0 - np.clip(ranks, 0.0, 1.0)
        return conf
