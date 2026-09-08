"""
================================================================================
Applied Paper - Step 0d: Primary Analysis Variable Upgrade (primary_arm_v2)
================================================================================
step0c's secondary_detail values that clear MIN_CELL_SIZE (in this
dataset: single_type1=42, multi_1+6=22, multi_1+2=9) are promoted directly
to levels of a single categorical variable, primary_arm_v2, instead of
collapsing to a binary single-vs-multi split. Nothing about which labels
survive is hardcoded: the promotion is driven entirely by MIN_CELL_SIZE
applied to this run's data, so the same script produces the correct
number of levels on a re-run with different data.

Note on interpretation (see the D-step crosstab): "multi_1+6" customers
are overwhelmingly type1-dominant by cumulative spend (definition b), so
they read as "type1 + type6 trial," whereas "multi_1+2" customers are
majority type2-dominant by spend — a genuinely different pattern. Both
labels are kept as-is (not renamed to imply one story) and the asymmetry
is reported as a descriptive footnote rather than smoothed over.

Output: reframe_output/analysis_ready_sample_v2.csv
        reframe_output/primary_arm_v2_cell_sizes.csv
        reframe_output/primary_arm_v2_manifest.json
================================================================================
"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("AD_DATA_ROOT", "./data"))
OUT_DIR = Path(os.environ.get("REFRAME_OUT", str(ROOT / "reframe_output")))
MIN_CELL_SIZE = int(os.environ.get("MIN_CELL_SIZE", "8"))

sample = pd.read_csv(OUT_DIR / "analysis_ready_sample.csv")
sample["registration_date"] = pd.to_datetime(sample["registration_date"])
sample["first_active_date"] = pd.to_datetime(sample["first_active_date"])

POOLED_LABELS = {"single_other_small_n", "multi_other_small_n"}


def _promote(row):
    detail = row["secondary_detail"]
    if pd.isna(detail) or detail in POOLED_LABELS:
        return "other"
    return f"single_type{detail}" if row["primary_arm"] == "single_channel_start" else f"multi_{detail}"


sample["primary_arm_v2_raw"] = sample.apply(_promote, axis=1)
raw_counts = sample["primary_arm_v2_raw"].value_counts()
print("Promoted (pre-reverification) distribution:")
print(raw_counts.to_string())

# Re-verify cell sizes dynamically (should be a no-op, but re-checked for re-run safety)
non_other = raw_counts.drop(labels=["other"], errors="ignore")
kept_levels = set(non_other[non_other >= MIN_CELL_SIZE].index)
sample["primary_arm_v2"] = np.where(sample["primary_arm_v2_raw"].isin(kept_levels),
                                     sample["primary_arm_v2_raw"], "other")

final_counts = sample["primary_arm_v2"].value_counts().rename_axis("level") \
    .reset_index(name="n_customers").sort_values("n_customers", ascending=False)
print(f"\nFinal level count: {len(final_counts)}")
print(final_counts.to_string(index=False))
final_counts.to_csv(OUT_DIR / "primary_arm_v2_cell_sizes.csv", index=False)

crosstab_v2 = pd.crosstab(sample["primary_arm_v2"], sample["first_campaign_type_b"], dropna=False)
print("\nprimary_arm_v2 x first_campaign_type_b crosstab (interpretation check):")
print(crosstab_v2.to_string())

# Reference category = largest level, chosen dynamically each run
reference_level = final_counts.iloc[0]["level"]
print(f"\nAuto-selected regression reference category: '{reference_level}'")
ordered = [reference_level] + [lv for lv in final_counts["level"] if lv != reference_level]
sample["primary_arm_v2"] = pd.Categorical(sample["primary_arm_v2"], categories=ordered)

sample.to_csv(OUT_DIR / "analysis_ready_sample_v2.csv", index=False)
with open(OUT_DIR / "primary_arm_v2_manifest.json", "w") as f:
    json.dump({
        "min_cell_size": MIN_CELL_SIZE, "n_total": int(len(sample)),
        "n_levels_final": len(final_counts),
        "level_counts": final_counts.set_index("level")["n_customers"].to_dict(),
        "reference_level_auto_selected": reference_level,
        "crosstab_def_b_by_level": crosstab_v2.to_dict(),
    }, f, indent=2, default=str)
print(f"\nSaved analysis_ready_sample_v2.csv. Next: step1_omnibus_and_pairwise.py")
