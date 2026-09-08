"""
================================================================================
Applied Paper - Step 0c: Primary Analysis Variable Finalization
================================================================================
Confirms the primary comparison variable (primary_arm). Single-channel
starters and co-first (multi-channel) starters are broken down internally
by campaign type / type-combination, and any level below MIN_CELL_SIZE is
pooled into "single_other_small_n" / "multi_other_small_n" rather than
dropped outright (individual cases are preserved for inspection).

Output: reframe_output/analysis_ready_sample.csv
        reframe_output/cofirst_combo_breakdown.csv
        reframe_output/definition_b_cell_size_check.csv
        reframe_output/cell_size_decisions_manifest.json
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

labels = pd.read_csv(OUT_DIR / "first_channel_labels.csv")
labels["registration_date"] = pd.to_datetime(labels["registration_date"])
labels["first_active_date"] = pd.to_datetime(labels["first_active_date"])
labels["is_cofirst_a"] = labels["is_cofirst_a"].astype(bool)
labels["large_reg_active_gap_flag"] = labels["large_reg_active_gap_flag"].fillna(False).astype(bool)

labels["primary_arm"] = np.where(labels["is_cofirst_a"], "multi_channel_start", "single_channel_start")
print(labels["primary_arm"].value_counts().to_string())

# --- single-starter breakdown by type ---
single_df = labels[labels["primary_arm"] == "single_channel_start"].copy()
tc_single = single_df["first_campaign_type_a"].value_counts().rename_axis("campaign_type") \
    .reset_index(name="n_customers").sort_values("campaign_type")
tc_single["below_min_cell_size"] = tc_single["n_customers"] < MIN_CELL_SIZE
kept_types_single = set(tc_single.loc[~tc_single["below_min_cell_size"], "campaign_type"])
single_df["single_type_detail"] = np.where(
    single_df["first_campaign_type_a"].isin(kept_types_single),
    single_df["first_campaign_type_a"].astype(int).astype(str), "single_other_small_n")
print(f"\nSingle-starter kept types: {sorted(kept_types_single)}")

# --- multi-starter breakdown by combo ---
multi_df = labels[labels["primary_arm"] == "multi_channel_start"].copy()
multi_df["cofirst_types_a"] = multi_df["cofirst_types_a"].fillna("").astype(str)
multi_df["combo_label_raw"] = multi_df["cofirst_types_a"].apply(
    lambda s: "+".join(sorted(x.strip() for x in s.split(",") if x.strip())))
combo_counts = multi_df["combo_label_raw"].value_counts().rename_axis("combo") \
    .reset_index(name="n_customers").sort_values("n_customers", ascending=False)
combo_counts["below_min_cell_size"] = combo_counts["n_customers"] < MIN_CELL_SIZE
kept_combos = set(combo_counts.loc[~combo_counts["below_min_cell_size"], "combo"])
multi_df["combo_detail"] = np.where(multi_df["combo_label_raw"].isin(kept_combos),
                                     multi_df["combo_label_raw"], "multi_other_small_n")
combo_counts.to_csv(OUT_DIR / "cofirst_combo_breakdown.csv", index=False)
print(f"\nMulti-starter kept combos: {sorted(kept_combos)}")
print(combo_counts.to_string(index=False))

# --- definition (b) cell-size check (no co-first, since it's a single-winner definition) ---
tc_b = labels["first_campaign_type_b"].value_counts(dropna=True).rename_axis("campaign_type") \
    .reset_index(name="n_customers").sort_values("campaign_type")
tc_b["below_min_cell_size"] = tc_b["n_customers"] < MIN_CELL_SIZE
tc_b.to_csv(OUT_DIR / "definition_b_cell_size_check.csv", index=False)

crosstab_ab = pd.crosstab(labels["primary_arm"], labels["first_campaign_type_b"], dropna=False)
print("\nprimary_arm x first_campaign_type_b crosstab:")
print(crosstab_ab.to_string())

final = labels.merge(single_df[["customer_id", "single_type_detail"]], on="customer_id", how="left") \
    .merge(multi_df[["customer_id", "combo_label_raw", "combo_detail"]], on="customer_id", how="left")
final["secondary_detail"] = np.where(final["primary_arm"] == "single_channel_start",
                                      final["single_type_detail"], final["combo_detail"])
final.to_csv(OUT_DIR / "analysis_ready_sample.csv", index=False)

manifest = {
    "min_cell_size": MIN_CELL_SIZE, "n_total": int(len(labels)),
    "primary_arm_counts": labels["primary_arm"].value_counts().to_dict(),
    "single_type_kept": sorted(int(t) for t in kept_types_single),
    "multi_combo_kept": sorted(kept_combos),
    "n_large_reg_active_gap_flagged": int(labels["large_reg_active_gap_flag"].sum()),
    "note": ("Primary comparison uses secondary_detail promoted to a 4-level categorical "
             "(single_type1 / multi_1+6 / multi_1+2 / other) in step0d, since each individual "
             "type/combo above cleared MIN_CELL_SIZE without needing the coarser binary split."),
}
with open(OUT_DIR / "cell_size_decisions_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2, default=str)
print("\nSaved analysis_ready_sample.csv. Next: step0d_arm_v2_promotion.py")
