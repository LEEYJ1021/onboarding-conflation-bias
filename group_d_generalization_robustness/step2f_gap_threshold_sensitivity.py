"""
================================================================================
Gap-Day Threshold Sensitivity Analysis
================================================================================
Purpose:
    The born-treated classification used a fixed threshold
    (BORN_TREATED_GAP_THRESHOLD = 1 day). A natural reviewer objection is
    "why 1 day and not some other cutoff?" This script re-derives the
    born-treated ratio under thresholds {0, 1, 3, 7} days, using the
    per-type diagnostic tables already produced by
    step2e_all_types_generalization.py (no re-running of the upstream
    pipeline is needed, since gap_days and is_valid_stack are threshold-
    independent quantities computed once).

    For each threshold we also re-run a chi-square test of homogeneity
    across campaign types, to check whether the "this is a structural,
    type-agnostic phenomenon" conclusion survives threshold choice.

Input:  step2_treatment_output/all_types/born_treated_diagnosis_type{T}.csv
        (produced by step2e_all_types_generalization.py)
Output: step2_treatment_output/all_types/gap_threshold_sensitivity_summary.csv
        step2_treatment_output/all_types/gap_threshold_sensitivity_summary.json
        step2_treatment_output/all_types/gap_threshold_sensitivity_by_type.png
================================================================================
"""
import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("AD_DATA_ROOT", "./data"))
STEP2_DIR = Path(os.environ.get("STEP2_OUT", str(ROOT / "step2_treatment_output")))
ALL_TYPES_DIR = STEP2_DIR / "all_types"

if not ALL_TYPES_DIR.exists():
    raise FileNotFoundError(
        f"{ALL_TYPES_DIR} not found — run step2e_all_types_generalization.py first."
    )

GAP_THRESHOLDS = [int(x) for x in os.environ.get("GAP_THRESHOLDS", "0,1,3,7").split(",")]
STABILITY_RANGE_MAX = float(os.environ.get("STABILITY_RANGE_MAX", "0.15"))

FILE_PATTERN = re.compile(r"^born_treated_diagnosis_type(\d+)\.csv$")
detail_files = {int(m.group(1)): f for f in ALL_TYPES_DIR.glob("born_treated_diagnosis_type*.csv")
                if (m := FILE_PATTERN.match(f.name))}
if not detail_files:
    raise FileNotFoundError(f"No born_treated_diagnosis_type*.csv found in {ALL_TYPES_DIR}")

print(f"Testing thresholds: {GAP_THRESHOLDS}")
print(f"Detected campaign types: {sorted(detail_files.keys())}\n")

type_details = {T: pd.read_csv(p) for T, p in detail_files.items()}

rows = []
for threshold in GAP_THRESHOLDS:
    for T, df in type_details.items():
        n_total = len(df)
        if n_total == 0:
            continue
        is_born = df["gap_days"] <= threshold
        n_born = int(is_born.sum())
        rows.append({
            "gap_threshold_days": threshold, "campaign_type": T,
            "n_total_adopters": n_total, "n_born_treated": n_born,
            "n_true_switcher": n_total - n_born,
            "born_treated_ratio": n_born / n_total,
            "n_valid_stack_total": int(df["is_valid_stack"].sum()) if "is_valid_stack" in df.columns else np.nan,
        })
sens_df = pd.DataFrame(rows).sort_values(["gap_threshold_days", "campaign_type"]).reset_index(drop=True)
print(sens_df.to_string(index=False))

# --- chi-square homogeneity test per threshold ---
homogeneity_rows = []
for threshold, g in sens_df.groupby("gap_threshold_days"):
    cont_table = g.set_index("campaign_type")[["n_born_treated", "n_true_switcher"]].T
    if cont_table.shape[1] < 2:
        continue
    chi2, pval, dof, _ = sps.chi2_contingency(cont_table.T)
    homogeneous = pval >= 0.05
    print(f"\nthreshold={threshold}d: mean ratio={g['born_treated_ratio'].mean():.1%}, "
          f"chi2={chi2:.3f}, p={pval:.4f} -> {'homogeneous' if homogeneous else 'heterogeneous'}")
    homogeneity_rows.append({"gap_threshold_days": threshold, "chi2": chi2, "pvalue": pval,
                              "homogeneous_at_05": homogeneous})
homogeneity_df = pd.DataFrame(homogeneity_rows)

# --- stability verdict: how much does the ratio move across thresholds, per type? ---
ratio_range = sens_df.groupby("campaign_type")["born_treated_ratio"].agg(["min", "max"])
ratio_range["range"] = ratio_range["max"] - ratio_range["min"]
max_range = ratio_range["range"].max()
stability_verdict = "STABLE" if max_range <= STABILITY_RANGE_MAX else "SENSITIVE"
print(f"\nMax variation across thresholds (any type): {max_range:.1%} -> {stability_verdict}")

sens_df.to_csv(ALL_TYPES_DIR / "gap_threshold_sensitivity_summary.csv", index=False)
with open(ALL_TYPES_DIR / "gap_threshold_sensitivity_summary.json", "w") as f:
    json.dump({
        "gap_thresholds_tested": GAP_THRESHOLDS,
        "stability_verdict": stability_verdict,
        "max_range_any_type": float(max_range),
        "ratio_range_by_type": ratio_range.reset_index().to_dict("records"),
        "homogeneity_tests_by_threshold": homogeneity_df.to_dict("records"),
    }, f, indent=2, default=str)

fig, ax = plt.subplots(figsize=(8, 5))
for T in sorted(type_details.keys()):
    sub = sens_df[sens_df["campaign_type"] == T].sort_values("gap_threshold_days")
    ax.plot(sub["gap_threshold_days"], sub["born_treated_ratio"], marker="o", label=f"type{T}")
ax.axvline(1, color="gray", linestyle="--", linewidth=1, label="adopted threshold (1 day)")
ax.set_xlabel("Gap-day threshold (days)")
ax.set_ylabel("Born-treated ratio")
ax.set_title("Gap-Day Threshold Sensitivity by Campaign Type")
ax.set_ylim(0, 1)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(ALL_TYPES_DIR / "gap_threshold_sensitivity_by_type.png", dpi=150)
plt.close()

print(f"\nSaved: gap_threshold_sensitivity_summary.csv/.json, gap_threshold_sensitivity_by_type.png")
