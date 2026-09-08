"""
================================================================================
Applied Paper - Step 2a: Diagnose primary_arm_v2 x reg_month Cell Sizes
================================================================================
step2's reg_month interaction Wald tests were uniformly demoted to
exploratory_only due to "does not have full rank" warnings. This script
pinpoints exactly which (arm, reg_month) cells are empty or sparse, so the
rank-deficiency is documented as a data-structure fact rather than left as
an opaque warning in the step2 log.

Output: reframe_output/step2a_cell_size_arm_x_regmonth.csv
        reframe_output/step2a_regmonth_diagnosis.json
================================================================================
"""
import os
import json
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("AD_DATA_ROOT", "./data"))
OUT_DIR = Path(os.environ.get("REFRAME_OUT", str(ROOT / "reframe_output")))
MIN_CELL_SIZE = int(os.environ.get("MIN_CELL_SIZE", "8"))

outcomes_all = pd.read_csv(OUT_DIR / "step1_outcomes_by_window.csv")
cust_meta = outcomes_all.drop_duplicates("customer_id")

arm_size = cust_meta["primary_arm_v2"].value_counts()
level_order = list(arm_size.index)
n_regmonth_levels = cust_meta["reg_month"].nunique(dropna=True)
print(f"primary_arm_v2 levels: {arm_size.to_dict()}")
print(f"reg_month levels: {n_regmonth_levels}\n")

cross = pd.crosstab(cust_meta["primary_arm_v2"], cust_meta["reg_month"]).reindex(level_order)
print(cross.to_string())
cross.to_csv(OUT_DIR / "step2a_cell_size_arm_x_regmonth.csv")

n_cells_empty = int((cross.values == 0).sum())
n_cells_below = int((cross.values < MIN_CELL_SIZE).sum())
print(f"\nEmpty cells: {n_cells_empty}/{cross.size} | Below MIN_CELL_SIZE: {n_cells_below}/{cross.size}")

empty_cells = [{"primary_arm_v2": a, "reg_month": str(m), "n": 0}
               for a in cross.index for m in cross.columns if cross.loc[a, m] == 0]
if empty_cells:
    print(f"\nEmpty cells (these force rank deficiency in the interaction design matrix):")
    for c in empty_cells:
        print(f"  {c['primary_arm_v2']} x reg_month={c['reg_month']}: n=0")

arm_effective_levels = (cross > 0).sum(axis=1)
print("\nEffective reg_month levels per arm (nonzero cells):")
for arm in level_order:
    n_eff = int(arm_effective_levels.get(arm, 0))
    flag = "" if n_eff >= n_regmonth_levels else "  <- fewer than the full level count"
    print(f"  {arm}: {n_eff} months with data{flag}")

with open(OUT_DIR / "step2a_regmonth_diagnosis.json", "w") as f:
    json.dump({
        "min_cell_size": MIN_CELL_SIZE, "n_regmonth_levels": int(n_regmonth_levels),
        "level_order": level_order, "n_cells_empty": n_cells_empty, "n_cells_below_min": n_cells_below,
        "empty_cells": empty_cells,
        "arm_effective_regmonth_levels": {k: int(v) for k, v in arm_effective_levels.items()},
    }, f, indent=2, default=str)
print(f"\nSaved step2a_cell_size_arm_x_regmonth.csv, step2a_regmonth_diagnosis.json")
