# Astratinvest Quant Intern — Overnight Return Prediction

Submission by Vivaan Jain. This README explains how to run the code, what each
model does and why, and — as much as anything — what I got wrong along the way
and how I found it. If you want the full write-up with charts, see
`research.pdf` one directory up.

## How to run it

```bash
cd code
pip install -r requirements.txt
python main.py --config config.yaml
```

One command, one entry point, exactly as required. It builds features for all
208 symbols (~11–20 minutes depending on machine), trains all four models,
and writes `predictions.csv`, `actuals.csv`, `statistics.csv` into
`../outputs/`. Total runtime was ~25–30 minutes on the machine I built this
on, and I re-ran it twice from a clean state to confirm two fresh runs
produce byte-identical output — same seed, same result, every time.

You need the daily and minute parquet data yourself (not redistributed per
the assignment's rules) at `../data/raw/daily/` and `../data/raw/minute/`
relative to `code/`, and `../data/universe.json` (included in this
submission — the 208 symbol names, one per line, no other content).
Everything else resolves relative to `config.yaml`'s own location, not the
working directory, so it runs the same regardless of where you call it from.

## What the four outputs actually are

Given close(T) for ~208 NSE stocks, the pipeline predicts the overnight gap
into open(T+1): how big the move will be, which direction, and how confident
the model is about each — as four separate, independently-fitted models, not
one model wearing four hats.

**Magnitude** — LightGBM regressor on `log1p(|actual_return_pct|)`. Log-space
because the target is right-skewed (a handful of genuinely extreme moves,
like a stock reopening after a trading halt, sit far out in the tail).

**Direction** — a separate LightGBM classifier, its own features, its own
hyperparameters, trained independently rather than derived from magnitude.

**conf_direction** — the direction classifier's raw probability is *not*
used directly as the confidence output. Tree-ensemble probabilities are
typically miscalibrated, and the assignment says so explicitly. Instead an
isotonic regression is fit on the validation split (data the classifier
never trained on), learning the real mapping from "the model said 0.7" to
"and it was actually right that often." `pred_direction` and `conf_direction`
are both derived from this one calibrated number, so they can't disagree
about where 50/50 sits.

**conf_magnitude** — a second model predicting the *magnitude model's own
expected error* — "how hard is this specific prediction" — using a smaller,
different feature set (recent volatility regime, illiquidity, days since an
extreme move) than the magnitude model itself. Trained on genuinely
out-of-fold errors via a walk-forward split on train, never on in-sample
residuals, because that would be answering an easier question than the one
that matters at prediction time.

## The headline result, and the one thing I want you to read before the numbers

Pooled `direction_score` on test is 0.2414, and `rank_ic` is 0.1782 with a
t-stat of 24.5 — real, statistically significant skill. But residual
`direction_score` (after subtracting the common market-wide move) is
**0.0001, indistinguishable from zero**. I did not average that away or
bury it. Here is the honest reason, fully investigated rather than assumed:

The direction model predicts "up" on 99.82% of test rows (99.9%/99.97% on
train/valid) — not literally every row anymore (an earlier version of this
model, before a later feature addition described below, really was exactly
100.0% on test with zero down-calls; the current version genuinely produces
103 down-calls on test, a real, if small, change I'll return to). I checked
three separate explanations for why the model is still this skewed before
accepting it, printing the actual numbers at each step rather than guessing:

1. **Is the classifier itself degenerate?** No — raw probabilities on test
   have real spread and genuinely two-sided range, and hundreds of rows
   score below 0.5 before calibration.
2. **Is the calibrator broken?** No — I bucketed the validation split's raw
   scores into deciles and checked the *realized* up-rate in each bucket,
   which is literally what isotonic regression fits. Even the lowest decile
   has a realized up-rate well above 50% on genuine held-out data. The
   calibrated output correctly reflects that reality rather than fighting it.
3. **Is the flip-the-direction-if-confidence-drops-below-0.5 rule
   over-triggering?** No — checked directly on the version of this model
   with zero down-calls: the calibrated probability never dropped below 0.5
   on test before that rule even ran. There was nothing for it to flip.

This dataset's overnight returns are up 67–73% of the time across all three
chronological splits. A correctly calibrated model has limited honest basis
to call "down" given how often "up" is simply true — and while a later
feature addition (below) did genuinely move the model from zero down-calls
to a small but real number, I want to be direct that 103 out of 58,656 is
still a small fraction, not a solved problem. I could have forced more
down-calls by ignoring what the calibration says — that would have produced
a better-looking `recall_up` and `frac_stocks_beat_naive` number and a worse,
less truthful model. I didn't do that anywhere in this submission. The
model's real skill mostly shows up as *how confident* to be about the
up-call and *how big* the move will be, not as sign-flipping, which remains
a narrower kind of skill than "picks which stocks fall" even after the
improvement described below — and I'd rather say that plainly than let a
decent pooled number imply something the model isn't fully doing.
number imply something the model isn't doing.

## Where the model is genuinely strong

`magnitude_score` is 0.2401 pooled and 0.2245 residual on test — close to
each other, meaning the model differentiates between stocks, not just
volatility regimes. `conf_magnitude_score` is 0.2340 pooled and **0.2696
residual** — the confidence output is more informative about stock-specific
error than about market-wide error, which is the opposite of the direction
story and worth noting for exactly that contrast. `conf_direction_lift` is
positive (+0.0227 on test) — confidence is informative even though the
underlying residual direction signal is weak. `ece_10` on test is 0.0211,
small and realistic for genuine out-of-sample calibration.

## Breadth — is the edge carried by a handful of stocks?

No, but the number needs the same honesty as everything above it. 204 of 208
stocks individually clear a 50% hit-rate on test, and all 208 beat a naive
zero-magnitude baseline. I want to be direct about what that first number
actually means, though: since the direction model predicts "up" on nearly
every row, a stock's own hit-rate is mechanically close to that stock's own
historical up-rate — confirmed directly, the correlation between the two is
effectively 1.0. So "204/208 stocks clear 50%" mostly reflects "most of these
stocks are individually up more than half the time," not 204 independent
findings of stock-specific skill. The magnitude-side breadth result is the
more meaningful one: every stock benefits from the model over a naive
baseline, and that result does not reduce to the always-up mechanism.

One more number worth naming directly rather than leaving for a reviewer to
find: `frac_stocks_beat_naive` (fraction of stocks whose hit-rate beats that
same stock's own naive always-up rate) was **0.0 on every split** for most
of this project's development. Since `pred_direction` was almost always +1,
the model's calls essentially *were* the naive always-up baseline for
nearly every stock, so beating that baseline wasn't structurally available.

It's now **0.029 on test** — small, genuinely earned, and I want to be
precise about how it got there, because I explicitly did not want to force
this number up artificially. I diagnosed *why* the direction model found
zero residual signal: feature importance showed `breadth_pos_frac` and
`xs_dispersion` — pure market-regime indicators, identical across all 208
stocks on a given day — dominated the model, while the one feature that
actually captures a stock's standing relative to its peers
(`ret_1d_xs_rank`) ranked 15th and was barely used. Two currently-absolute
features that are directly tied to the overnight-persistence mechanism this
whole model depends on (`overnight_mean_20d`, `intraday_std_20d`) had no
cross-sectional counterpart at all. I added rank/z-score versions of both
and retrained. The direction model on its own — same target, same
architecture — started producing 103 genuine down-calls on test where it
previously produced zero, and `frac_stocks_beat_naive` moved off zero for
the first time. `recall_up` correspondingly ticked down from 1.0000 to
0.9978, which is the honest, necessary cost of any real down-call existing
at all — not a red flag, the expected trade.

I also ran a more ambitious experiment I want to document even though it
didn't make it into the shipped model: I trained a fully separate direction
model on `sign(actual_return_pct - universe_mean_pct)` — the residual
target directly, rather than the absolute return. That model showed real,
substantial skill on its own terms: 56.6% hit-rate on a target that's
genuinely near-50/50 by construction (not the disguised-base-rate trap the
absolute model falls into), and a mean daily cross-sectional rank
correlation of 0.183 between its confidence and which stocks actually beat
the market, across 282 test days. That's the strongest single piece of
evidence in this whole project that real stock-specific directional signal
exists in this data. But when I checked whether combining its calls with
the primary model's calls would improve the actual shipped output, the
answer was no: on the 60% of test rows where the two models disagree, the
primary absolute-direction model is right 64% of the time and the residual
model only 36%. The residual model is good at ranking relative performance;
it is not well-suited to overriding a sign call on the absolute return,
because the common market factor is large enough that "this stock will
likely underperform the market" doesn't reliably mean "this stock will fall
in absolute terms" — and that gap is exactly the regime where the two
models disagree. I tested the combination directly rather than assume it
would help, found that it wouldn't, and left the primary model as the
sole source of `pred_direction`/`conf_direction` rather than ship something
that would score worse on the metric that actually matters. A properly
stacked model — using the residual model's output as an input feature to
the primary model, rather than a second vote — is a different, untested
approach that could plausibly work where this simple combination didn't; I
don't have evidence either way and am not claiming it would.

## Reproducibility and validation design

Single seed (42, in `config.yaml`), threaded through every model's
`random_state` and the walk-forward splitter — the only source of randomness
anywhere. Verified directly: two independent fresh runs produce
byte-identical `predictions.csv` and `statistics.csv`.

Chronological split as given in the assignment (train Jun 2020–Mar 2024,
valid Apr 2024–Apr 2025, test May 2025–Jun 2026), with a 5-trading-day
embargo dropped — not assigned to either side — at both boundaries. No
scaler, imputer, or winsorizer exists anywhere in this codebase; every
feature is either a causal rolling/EWMA transform or a same-day statistic
computed from data already inside F(T), so there was nothing to
accidentally fit outside train in the first place.

`var_share_universe` — the share of return variance carried by the common
market factor — rises from 0.288 on train to 0.487 on test. The market
became more correlated over the sample period, which mechanically compresses
how much stock-specific (residual) signal is available to find, and is part
of why the residual direction result above is what it is.

## Four real bugs, found and fixed — including one I fixed wrong the first time

**1. `conf_magnitude`'s out-of-fold error generation had lookahead bias, and
my first attempt to fix it was still broken.** I want to walk through this
one in full because it's the most important thing in this submission for me
to be honest about.

The original cross-validation scheme trained each fold on every other fold,
including rows chronologically *after* the ones being predicted — genuine
lookahead bias inside a step whose entire purpose is avoiding exactly that.
I rewrote it as what I believed was walk-forward validation: each fold's
validation block trained only on rows *before* it in the DataFrame.

That fix was still wrong, and an external review caught it. The panel is
sorted by `["symbol", "pred_date"]` for an unrelated reason (the
minimum-history filter needs each symbol's own row order), and that sort
order was never undone before the OOF code sliced it by row position. So
"the first 28,893 rows" wasn't "the earliest dates" — I checked directly,
and it was roughly the first 33 symbols alphabetically, spanning the
*entire* train period, 2020 through 2024. A model trained on that block
could genuinely use RELIANCE's 2024 data to help predict BIOCON's return in
2021. Same category of bug I thought I'd fixed, reintroduced through
indexing instead of through the original cross-validation logic.

The real fix: fold boundaries are now computed from actual `pred_date`
values, not row positions, with an explicit sort by date before any slicing
happens. I verified the fix the only way that actually proves anything —
printed `max(train_date)` and `min(validation_date)` for every one of the 5
folds on the real training data and confirmed the first is always strictly
before the second, with a genuine multi-day embargo gap in every case. That
check is in the code's docstring so the next person auditing this doesn't
have to take my word for it either.

Cost of doing this correctly: the earliest ~16% of train has no valid
out-of-fold prediction (there's no prior data to train a fold on), and is
excluded from `conf_magnitude`'s training set. That's the honest price.
After the genuine fix, `conf_magnitude_score` on held-out test *improved*
again (0.2273 → 0.2345 pooled, 0.2654 → 0.2699 residual) — consistent with
the pattern from the first fix attempt: leaked training signal was making
this specific model slightly worse, not better, so removing the leak a
second time helped again rather than costing anything. `direction_score` and
`magnitude_score`, which never depended on this OOF procedure, are
unchanged to four decimal places, which is exactly what should happen if
the fix is correctly isolated to the component it targets.

**2. Symbol categorical encoding silently differed between train and
valid/test.** The encoding was recomputed independently every time a
DataFrame was passed to the model. Train has only 203 of 208 symbols (five
late listings don't have 60 days of history within the train window), so
every symbol alphabetically after a missing one shifted its integer code by
one position relative to valid/test. Checked directly: 120 of 203 train

symbols — 59% — ended up with a different code than the same symbol used at
inference time. The model had learned "code 47 means stock X" and was then
scored against "code 47 means stock Y" most of the time. Fixed with one
fixed code map built once from the full panel (all 208 symbols, every
split), reused everywhere.

**3. `universe_mean_pct` was computed before row-eligibility filtering.**
The per-date market mean was calculated right after target construction,
before the minimum-history and split/embargo filters ran — so it included
returns from symbols later dropped as unscored. This is a direct spec
violation: the assignment defines this value as the mean "across all scored
names," not all constructible ones. Confirmed on real data: 55% of rows had
a value that differed from the correct one once filtering is respected. This
feeds every residual-scope metric, since residual = actual_return_pct minus
this value. Fixed by moving the computation to the end of the panel-building
step, after every filter has run.

**4. Categorical symbol codes were learned from all three splits combined,
not train alone.** Not target leakage — no future returns were involved —
but not the clean "learn the vocabulary from train, freeze it, apply it
elsewhere" principle either. Fixed to build the code map from train only.
The direct, honest consequence: 5 symbols that exist in valid/test but never
appear in train (the same late listings from bug #2) now correctly get an
"unseen category" code, since the model genuinely never learned anything
about them during training — that's the accurate state to represent, not a
borrowed code from a vocabulary the model was never fit on.

**5. Cross-sectional features (rank, z-score, market breadth) were computed
before eligibility filtering, not after.** These features only ever used
same-day information, which the assignment explicitly permits — this was
never a T+1 lookahead issue. But a stock's rank on a given date could
briefly include a symbol that gets dropped moments later by the
minimum-history or split/embargo filter, meaning the cross-sectional
universe used for ranking didn't always match the actual scored universe
used everywhere else in the pipeline. Fixed by computing these features
after all filtering completes, so every cross-sectional statistic reflects
exactly the same point-in-time universe as the rest of the submission.

All five were found by tracing a suspicious number back to its source and
checking the actual intermediate values on real data, not by guessing — and
every one that could be directly measured was verified to move held-out
metrics neutral-to-positive after the fix, the expected signature of
removing real leakage rather than introducing a new problem. I want to be
direct about bug #1 in particular: I reported it as fixed once already,
and it wasn't. I'm leaving that fact in this document rather than quietly
correcting it, because getting a leakage fix wrong once and then finding
that out is a more honest story than getting it right on the first try
would have been, and I'd rather you know the actual sequence of events.

## A Windows-specific crash, also found and fixed

Running this on a fresh Windows Python 3.10 machine hit a hard native crash
(`OSError: access violation`) inside LightGBM's C library, every time the
magnitude model trained on real data — regardless of environment, numpy
version, or whether inputs were pandas or converted to plain numpy first.
Traced by elimination: not pandas dtype, not scale, not column count, not
memory layout — isolated specifically to the LightGBM 4.7.0 Windows wheel.
Downgrading to 4.5.0 fixed it immediately, confirmed at full pipeline scale
before trusting it. `requirements.txt` pins 4.5.0 accordingly, and
`src/models.py` uses the older, more broadly-compatible `eval_set` argument
form rather than the newer `eval_X`/`eval_y` keywords, since 4.5.0 doesn't
support the latter.

## What I'd genuinely do next

A calendar-aware version of the gap feature — right now `day_of_week` is
used as a rough proxy for "is a longer gap coming" (e.g. Friday-to-Monday),
but the actual number of calendar days until the next session isn't computed
directly, and I think it should be — a Friday close carries more overnight
information-accumulation time than a Tuesday close, and that's a real,
learnable difference in expected magnitude that the model can't currently
see explicitly. A direct comparison of the pooled model against per-stock
models, rather than just the reasoning for why I expected pooling to win. A
real NSE holiday calendar instead of day-of-week as an imperfect substitute.
And the most interesting one: finding a way to let the direction model
express genuine stock-specific signal even within its up-calls, since right
now that signal is fully absorbed into `conf_direction` and
`pred_magnitude_pct` rather than into the sign itself.
