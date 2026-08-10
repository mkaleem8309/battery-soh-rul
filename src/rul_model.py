"""
RUL (Remaining Useful Life) prediction with uncertainty bands.
Person D's part of the Battery SoH + RUL Estimator.

INPUT CONTRACT (from teammate's SoH model, outputs/soh_predictions.csv):
    cycle_id, cell_id, soh_predicted, soh_ground_truth

OUTPUT CONTRACT (what the rest of the team builds against):
    cell_id, rul_best_cycles, rul_likely_cycles, rul_worst_cycles,
    current_soh, trend_slope, top_driver, driver_importance_scores (dict)

Method (MVP, v1):
  1. Fit a degradation trend (linear, or exponential if it fits recent
     history meaningfully better) on the trailing RECENT_WINDOW cycles.
  2. Extrapolate forward to SOH_THRESHOLD to get rul_likely_cycles.
  3. Build a distribution of local decline rates from short rolling
     sub-windows of that same trailing history; rul_best_cycles uses a
     shallow (BEST_QUANTILE) decline rate, rul_worst_cycles uses a
     steep (WORST_QUANTILE) one. All three bands are anchored at the
     same (current_cycle, current_soh) point so they're directly comparable.
  4. Never emits a single point estimate -- best/likely/worst always
     come back together.

ASSUMPTION (stated explicitly, applies to every RUL number this module
produces): future usage continues similar to the trailing RECENT_WINDOW
cycles -- same charge/discharge pattern, no new stress events (e.g. fast
charging spikes, thermal excursions, sudden depth-of-discharge changes).
This is a "if nothing changes" projection, not a guarantee.

v2 (later, only if v1 works and there's spare time -- ask before switching):
  GradientBoostingRegressor(loss="quantile") at alpha=0.1/0.5/0.9, trained
  on residual/degradation features, for real quantile regression bands
  instead of the slope-quantile heuristic used here.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

# ---- tunables -------------------------------------------------------------
SOH_THRESHOLD = 80.0        # % SoH considered end-of-life
RECENT_WINDOW = 60          # trailing cycles used to fit the trend
SUBWINDOW_SIZE = 20         # size of each rolling sub-window for the slope distribution
SUBWINDOW_STRIDE = 5
BEST_QUANTILE = 0.20        # shallowest 20% of local decline rates -> optimistic case
WORST_QUANTILE = 0.80       # steepest 20% of local decline rates -> pessimistic case
MIN_CYCLES_REQUIRED = 10    # below this we don't trust a trend fit at all
MIN_SLOPE_MAGNITUDE = 0.005 # %/cycle -- below this, decline signal is indistinguishable
                             # from noise; don't extrapolate off it (avoids e.g. a
                             # near-zero slope producing a "1.3 million cycle" RUL)
MAX_REASONABLE_RUL = 5000   # cycles -- hard cap as a second line of defense
NEAR_EOL_MARGIN = 5.0       # percentage points of SoH above threshold
MAX_REASONABLE_RUL_NEAR_EOL = 500  # tighter cap for cells already close to threshold --
                             # a flat *recent* slope there is weak evidence of real
                             # safety margin, since noise or a short plateau can look
                             # identical to genuine stabilization this close to EOL

SOH_PREDICTIONS_PATH = "outputs/soh_predictions.csv"
RUL_OUTPUT_PATH = "outputs/rul_predictions.csv"


# ---- mock data (use only if the real SoH predictions aren't ready) --------
def make_mock_soh_dataframe(n_cycles=300, seed=42):
    """Hand-written mock SoH history matching the input contract:
    cycle_id, cell_id, soh_predicted, soh_ground_truth.
    5 cells with different degradation personalities to exercise the
    RUL logic against realistic edge cases (clean linear, exponential
    knee, noisy sensor, slow/healthy, late-life acceleration)."""
    rng = np.random.default_rng(seed)
    rows = []

    def add_cell(cell_id, soh_fn, noise_std):
        for c in range(1, n_cycles + 1):
            true_soh = soh_fn(c)
            ground_truth = true_soh + rng.normal(0, 0.15)
            predicted = true_soh + rng.normal(0, noise_std)
            rows.append((c, cell_id, predicted, ground_truth))

    add_cell("cell_01", lambda c: 100 - 0.05 * c, noise_std=0.2)                 # clean linear
    add_cell("cell_02", lambda c: 55 + 45 * np.exp(-0.008 * c), noise_std=0.25)  # exponential knee
    add_cell("cell_03", lambda c: 100 - 0.045 * c, noise_std=0.6)                # noisy linear
    add_cell("cell_04", lambda c: 100 - 0.015 * c, noise_std=0.2)                # slow/healthy
    def cell5(c):
        return 100 - 0.02 * c if c <= 250 else (100 - 0.02 * 250) - 0.35 * (c - 250)
    add_cell("cell_05", cell5, noise_std=0.25)                                   # late-life cliff

    return pd.DataFrame(rows, columns=["cycle_id", "cell_id", "soh_predicted", "soh_ground_truth"])


def load_soh_data():
    """Real predictions if the teammate's file exists and has enough
    history per cell, otherwise fall back to the mock so RUL work isn't
    blocked. Prints which source it used."""
    if os.path.exists(SOH_PREDICTIONS_PATH):
        df = pd.read_csv(SOH_PREDICTIONS_PATH)
        counts = df.groupby("cell_id")["cycle_id"].count()
        if (counts >= MIN_CYCLES_REQUIRED).all() and len(df) > 0:
            print(f"[rul_model] using real SoH predictions from {SOH_PREDICTIONS_PATH} "
                  f"({df['cell_id'].nunique()} cells)")
            return df
        print(f"[rul_model] {SOH_PREDICTIONS_PATH} exists but is too thin "
              f"(needs >= {MIN_CYCLES_REQUIRED} cycles/cell) -- using mock data instead")
    else:
        print(f"[rul_model] {SOH_PREDICTIONS_PATH} not found -- using mock data")
    return make_mock_soh_dataframe()


# ---- step 1: fit a degradation trend on recent history --------------------
def _linear_fit(x, y):
    slope, intercept, r, p, se = stats.linregress(x, y)
    pred = slope * x + intercept
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return {"type": "linear", "slope": slope, "intercept": intercept, "r2": r2}


def _exp_decay_fit(x, y):
    """SoH(c) = floor + (y0 - floor) * exp(-k * (c - c0))"""
    x0, y0 = x[0], y[0]

    def model(c, floor, k):
        return floor + (y0 - floor) * np.exp(-k * (c - x0))

    try:
        popt, _ = curve_fit(model, x, y, p0=[max(0, y.min() - 5), 0.01],
                             maxfev=5000, bounds=([0, 1e-6], [100, 1]))
        pred = model(x, *popt)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return {"type": "exponential", "floor": popt[0], "k": popt[1], "y0": y0, "x0": x0, "r2": r2}
    except Exception:
        return None


def fit_trend_for_cell(cell_df, window=RECENT_WINDOW):
    cell_df = cell_df.sort_values("cycle_id")
    recent = cell_df.tail(window)
    x = recent["cycle_id"].values.astype(float)
    y = recent["soh_predicted"].values.astype(float)

    lin = _linear_fit(x, y)
    exp = _exp_decay_fit(x, y)

    # exponential only wins if it's a meaningfully better fit -- otherwise
    # keep linear, since it's simpler and more stable to extrapolate for MVP
    best = lin
    if exp is not None and exp["r2"] > lin["r2"] + 0.02:
        best = exp
    best["window_start_cycle"] = int(x.min())
    best["window_end_cycle"] = int(x.max())
    best["n_points"] = len(x)
    return best


# ---- step 2: extrapolate to threshold for rul_likely_cycles ---------------
def _instantaneous_slope_at_current(fit, current_cycle):
    """Slope at the most recent cycle, for either fit type, so best/likely/worst
    all anchor off the same reference point."""
    if fit["type"] == "linear":
        return fit["slope"]
    floor, k, y0, x0 = fit["floor"], fit["k"], fit["y0"], fit["x0"]
    soh_at_c = floor + (y0 - floor) * np.exp(-k * (current_cycle - x0))
    return -k * (soh_at_c - floor)


def _cycles_from_slope(current_soh, slope, threshold=SOH_THRESHOLD):
    # tighter cap once the cell is already close to EOL -- a flat trend there
    # is weaker evidence of real headroom than the same flat trend far from EOL
    # (a short plateau or noise can look identical to genuine stabilization
    # this close to threshold, so we don't want to promise 5000 cycles off it)
    cap = MAX_REASONABLE_RUL_NEAR_EOL if (current_soh - threshold) <= NEAR_EOL_MARGIN else MAX_REASONABLE_RUL
    if slope >= -MIN_SLOPE_MAGNITUDE:
        # flat, improving, or too shallow to trust -- decline signal isn't
        # distinguishable from noise yet. Rather than emit NaN (which breaks
        # anything downstream expecting a number), treat it as "no meaningful
        # degradation detected yet" and report the cap -- i.e. at least this
        # many cycles remain under current conditions.
        return cap
    rul = (current_soh - threshold) / (-slope)
    rul = max(rul, 0.0)
    return min(rul, cap)  # cap absurd extrapolations from a barely-negative slope


# ---- step 3: best/worst bands from a local slope distribution -------------
def _rolling_slope_distribution(cell_df, window=RECENT_WINDOW,
                                 sub_size=SUBWINDOW_SIZE, stride=SUBWINDOW_STRIDE):
    cell_df = cell_df.sort_values("cycle_id")
    recent = cell_df.tail(window)
    x = recent["cycle_id"].values.astype(float)
    y = recent["soh_predicted"].values.astype(float)

    slopes = []
    for start in range(0, len(x) - sub_size + 1, stride):
        xs, ys = x[start:start + sub_size], y[start:start + sub_size]
        slope, *_ = stats.linregress(xs, ys)
        slopes.append(slope)
    return np.array(slopes) if slopes else np.array([np.nan])


def compute_bands_for_cell(cell_df, threshold=SOH_THRESHOLD):
    cell_df = cell_df.sort_values("cycle_id")
    fit = fit_trend_for_cell(cell_df)
    current_cycle = cell_df["cycle_id"].max()
    current_soh = float(cell_df.loc[cell_df["cycle_id"] == current_cycle, "soh_predicted"].values[0])

    slope_likely = _instantaneous_slope_at_current(fit, current_cycle)

    slopes = _rolling_slope_distribution(cell_df)
    declining = slopes[slopes < 0]
    if len(declining) == 0:
        declining = np.array([slope_likely])

    slope_best = np.quantile(declining, BEST_QUANTILE)
    slope_worst = np.quantile(declining, WORST_QUANTILE)
    if abs(slope_best) > abs(slope_worst):  # keep best = shallowest, worst = steepest
        slope_best, slope_worst = slope_worst, slope_best

    # slope_likely comes from a different fit (whole-window trend) than
    # slope_best/slope_worst (local sub-window quantiles), so nothing
    # guarantees it falls between them -- e.g. a near-flat whole-window
    # fit can hit the MAX_REASONABLE_RUL cap while noisier sub-windows
    # still show decline, producing worst < likely < best violations that
    # would confuse anyone consuming these bands. Clamp so the ordering
    # worst <= likely <= best always holds.
    slope_likely = min(max(slope_likely, slope_worst), slope_best)

    return {
        "cell_id": cell_df["cell_id"].iloc[0],
        "rul_best_cycles": _round_or_none(_cycles_from_slope(current_soh, slope_best, threshold)),
        "rul_likely_cycles": _round_or_none(_cycles_from_slope(current_soh, slope_likely, threshold)),
        "rul_worst_cycles": _round_or_none(_cycles_from_slope(current_soh, slope_worst, threshold)),
        "current_soh": round(current_soh, 2),
        "trend_slope": round(float(slope_likely), 5),
        # --- clean slot: filled in by merge_driver_scores() once the
        # driver-identification teammate's output is ready ---
        "top_driver": None,
        "driver_importance_scores": {},
    }


def _round_or_none(v):
    return None if v is None else round(float(v), 1)


# ---- step 4/5: assemble the output contract --------------------------------
OUTPUT_COLUMNS = [
    "cell_id", "rul_best_cycles", "rul_likely_cycles", "rul_worst_cycles",
    "current_soh", "trend_slope", "top_driver", "driver_importance_scores",
]


def build_rul_output(df, threshold=SOH_THRESHOLD):
    records = [compute_bands_for_cell(g, threshold) for _, g in df.groupby("cell_id")]
    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


FEATURE_IMPORTANCES_PATH = "outputs/feature_importances.csv"
PER_CELL_DRIVERS_PATH = "outputs/per_cell_drivers.csv"


def merge_driver_scores(rul_df, driver_df=None, on="cell_id"):
    """Merge in the driver-identification teammate's output.

    Handles two shapes, auto-detected from driver_df's columns:
      - per-cell (outputs/per_cell_drivers.csv, preferred when present):
        cell_id, top_driver, driver_importance_scores -- left-joined on `on`,
        so RUL rows are never dropped if a cell is missing driver data.
        driver_importance_scores may arrive as a JSON string (that's how
        it round-trips through CSV) -- parsed back into a dict here so the
        output contract (a dict, not a string) holds.
      - fleet-wide (outputs/feature_importances.csv): Feature, Importance
        columns with one row per feature and no cell_id at all. Used only
        as a fallback when no per-cell file exists yet.

    If driver_df is None, prefers PER_CELL_DRIVERS_PATH, falls back to
    FEATURE_IMPORTANCES_PATH. If nothing is available at all, rul_df is
    returned unchanged (top_driver stays None / scores stay {}) so this
    never breaks the pipeline.
    """
    if driver_df is None:
        if os.path.exists(PER_CELL_DRIVERS_PATH):
            driver_df = pd.read_csv(PER_CELL_DRIVERS_PATH)
        elif os.path.exists(FEATURE_IMPORTANCES_PATH):
            driver_df = pd.read_csv(FEATURE_IMPORTANCES_PATH)
        else:
            print(f"[rul_model] no driver output found at {PER_CELL_DRIVERS_PATH} or "
                  f"{FEATURE_IMPORTANCES_PATH} -- leaving top_driver / driver_importance_scores empty")
            return rul_df

    out = rul_df.drop(columns=["top_driver", "driver_importance_scores"]).copy()

    if {"cell_id", "top_driver", "driver_importance_scores"}.issubset(driver_df.columns):
        # per-cell shape -- parse driver_importance_scores back into a dict if it
        # came in as a JSON string (normal when read straight from CSV)
        driver_df = driver_df.copy()
        driver_df["driver_importance_scores"] = driver_df["driver_importance_scores"].apply(
            lambda v: json.loads(v) if isinstance(v, str) else v
        )
        merged = out.merge(driver_df[["cell_id", "top_driver", "driver_importance_scores"]],
                            on=on, how="left")
        return merged[OUTPUT_COLUMNS]

    if {"Feature", "Importance"}.issubset(driver_df.columns):
        # fleet-wide shape -- same ranking applied to every cell
        ranked = driver_df.sort_values("Importance", ascending=False)
        global_top_driver = ranked.iloc[0]["Feature"]
        global_scores = dict(zip(ranked["Feature"], ranked["Importance"].round(4)))
        out["top_driver"] = global_top_driver
        out["driver_importance_scores"] = [global_scores] * len(out)
        return out[OUTPUT_COLUMNS]

    print(f"[rul_model] driver_df columns {list(driver_df.columns)} don't match either "
          f"expected shape -- leaving top_driver / driver_importance_scores empty")
    return rul_df


if __name__ == "__main__":
    df = load_soh_data()
    out = build_rul_output(df)
    out = merge_driver_scores(out)  # fills top_driver / driver_importance_scores if available

    os.makedirs("outputs", exist_ok=True)
    out.to_csv(RUL_OUTPUT_PATH, index=False)
    print(f"\nSaved RUL predictions -> {RUL_OUTPUT_PATH}\n")

    pd.set_option("display.width", 140)
    print(out.to_string(index=False))

    print(f"\nASSUMPTION: future usage continues similar to the trailing {RECENT_WINDOW} "
          f"cycles (same charge/discharge pattern, no new stress events like fast-charging "
          f"spikes or thermal excursions). Threshold = {SOH_THRESHOLD}% SoH.")
