"""
================================================================================
Applied Paper - Step 2: Heterogeneity Analysis (Interaction Wald Tests)
================================================================================
Tests whether primary_arm_v2's association with outcomes is moderated by
structural, registration-time covariates only (device_type_mode, reg_month).
Spend-derived covariates are deliberately excluded: since every customer in
this sample is new, there is no pre-period, so "prior customer scale" is
not an identifiable concept here — this is a designed limitation, stated
up front, not a post-hoc excuse.

Robustness handling: cell-size gaps in the arm x covariate crosstab
(confirmed in step2a) commonly make the interaction design matrix rank-
deficient. Each Wald test is wrapped in try/except; a caught
"does not have full rank" warning or an outright failure demotes that
(outcome, window, covariate) combination to exploratory_only / not_estimable
status rather than crashing or silently reporting a spurious result.

Outcome scope follows step1's confirmed findings: log_spend_cum only at
its confirmed window (60 days); n_types_active_avg_post_day0 at all
windows; time_to_new_type_post_day0 is exploratory-only throughout since
its omnibus test was mostly non-significant in step1.

Output: reframe_output/step2_cell_size_arm_x_device.csv
        reframe_output/step2_interaction_wald_results.csv
        reframe_output/step2_interaction_coefficients.csv
        reframe_output/step2_manifest.json
================================================================================
"""
import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(os.environ.get("AD_DATA_ROOT", "./data"))
OUT_DIR = Path(os.environ.get("REFRAME_OUT", str(ROOT / "reframe_output")))
MIN_CELL_SIZE = int(os.environ.get("MIN_CELL_SIZE", "8"))
ALPHA = float(os.environ.get("ALPHA", "0.05"))
LOG_SPEND_CONFIRMED_WINDOW = int(os.environ.get("LOG_SPEND_CONFIRMED_WINDOW_DAYS", "60"))

outcomes_all = pd.read_csv(OUT_DIR / "step1_outcomes_by_window.csv")
arm_size = outcomes_all.drop_duplicates("customer_id")["primary_arm_v2"].value_counts()
level_order = list(arm_size.index)
reference_level = level_order[0]
print(f"primary_arm_v2 levels: {arm_size.to_dict()} | reference: '{reference_level}'\n")

has_device = "device_type_mode" in outcomes_all.columns
has_reg_month = "reg_month" in outcomes_all.columns
cust_meta = outcomes_all.drop_duplicates("customer_id")
use_device = has_device and cust_meta["device_type_mode"].nunique(dropna=True) >= 2
use_regmonth = has_reg_month and cust_meta["reg_month"].nunique(dropna=True) >= 2
print(f"device_type_mode interaction enabled: {use_device}")
print(f"reg_month interaction enabled: {use_regmonth}")

device_arm_ok = False
if use_device:
    cross = pd.crosstab(cust_meta["primary_arm_v2"], cust_meta["device_type_mode"])
    print("\narm x device_type_mode crosstab:\n" + cross.to_string())
    device_arm_ok = bool((cross.values < MIN_CELL_SIZE).sum() == 0)
    cross.to_csv(OUT_DIR / "step2_cell_size_arm_x_device.csv")
    if not device_arm_ok:
        print("Sparse/empty cells present -> device interaction demoted to exploratory_only.")

run_specs = [("log_spend_cum", LOG_SPEND_CONFIRMED_WINDOW, "confirmed")]
for w in sorted(outcomes_all["window_days"].unique()):
    run_specs.append(("n_types_active_avg_post_day0", int(w), "confirmed"))
for w in sorted(outcomes_all["window_days"].unique()):
    run_specs.append(("time_to_new_type_post_day0", int(w), "exploratory_only"))


def _build_formula(outcome, interaction_with):
    ref_term = f"C(primary_arm_v2, Treatment(reference='{reference_level}'))"
    terms = [f"{ref_term} * C({interaction_with})"]
    other = []
    if interaction_with != "device_type_mode" and use_device:
        other.append("C(device_type_mode)")
    if interaction_with != "reg_month" and use_regmonth:
        other.append("C(reg_month)")
    return f"{outcome} ~ " + " + ".join(terms + other)


wald_rows, coef_rows = [], []
for outcome, window_days, status in run_specs:
    sub_w = outcomes_all[outcomes_all["window_days"] == window_days].copy()
    for interaction_with, enabled in [("device_type_mode", use_device), ("reg_month", use_regmonth)]:
        if not enabled:
            continue
        formula = _build_formula(outcome, interaction_with)
        try:
            model = smf.ols(formula, data=sub_w).fit(cov_type="HC1")
        except Exception as e:
            print(f"  fit failed: {outcome} x {interaction_with} (w={window_days}): {e}")
            continue

        interaction_terms = [t for t in model.params.index if ":" in t and interaction_with in t]
        if not interaction_terms:
            continue

        cell_status = status
        if interaction_with == "device_type_mode" and use_device and not device_arm_ok:
            cell_status = "exploratory_only"

        hypothesis = " = 0, ".join(interaction_terms) + " = 0"
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                wald_result = model.wald_test(hypothesis, use_f=True)
                rank_deficient = any("does not have full rank" in str(w_.message) for w_ in caught)
            wald_p = float(np.ravel(wald_result.pvalue)[0])
            wald_f = float(np.ravel(wald_result.statistic)[0])
        except Exception as e:
            wald_rows.append({"outcome": outcome, "window_days": window_days, "interaction_with": interaction_with,
                               "wald_f_stat": np.nan, "wald_p_value": np.nan, "significant": False,
                               "status": "not_estimable", "n_obs": int(model.nobs)})
            print(f"  {outcome:<28} w={window_days:>2}d x {interaction_with:<16} Wald not estimable: {e}")
            continue

        if rank_deficient:
            cell_status = "exploratory_only"

        wald_rows.append({"outcome": outcome, "window_days": window_days, "interaction_with": interaction_with,
                           "wald_f_stat": wald_f, "wald_p_value": wald_p,
                           "significant": bool(wald_p < ALPHA and not rank_deficient),
                           "status": cell_status, "n_obs": int(model.nobs)})
        rank_note = " [rank-deficient -> exploratory]" if rank_deficient else ""
        print(f"  {outcome:<28} w={window_days:>2}d x {interaction_with:<16} "
              f"F={wald_f:.3f} p={wald_p:.4f} [{cell_status}]{rank_note}")

        for term in interaction_terms:
            coef_rows.append({"outcome": outcome, "window_days": window_days, "interaction_with": interaction_with,
                               "term": term, "coef": model.params[term], "se_hc1": model.bse[term],
                               "p_value": model.pvalues[term], "status": cell_status})

wald_df, coef_df = pd.DataFrame(wald_rows), pd.DataFrame(coef_rows)
wald_df.to_csv(OUT_DIR / "step2_interaction_wald_results.csv", index=False)
coef_df.to_csv(OUT_DIR / "step2_interaction_coefficients.csv", index=False)

manifest = {
    "min_cell_size": MIN_CELL_SIZE, "alpha": ALPHA,
    "log_spend_confirmed_window": LOG_SPEND_CONFIRMED_WINDOW,
    "level_order": level_order, "reference_level": reference_level,
    "use_device_interaction": use_device, "use_regmonth_interaction": use_regmonth,
    "device_arm_cell_size_ok": device_arm_ok if use_device else None,
    "n_wald_tests": int(len(wald_df)), "n_wald_significant": int(wald_df["significant"].sum()) if len(wald_df) else 0,
    "note_limitation": ("Heterogeneity is restricted to registration-time structural covariates. "
                         "'Prior customer scale' is not identifiable for this all-new-customer sample "
                         "and is not attempted — a designed constraint, not a post-hoc finding."),
}
with open(OUT_DIR / "step2_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2, default=str)
print(f"\nSaved step2_interaction_wald_results.csv, step2_interaction_coefficients.csv, step2_manifest.json")
