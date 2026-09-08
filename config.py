"""
================================================================================
config.py — Shared configuration and path resolution for the pipeline
================================================================================
All pipeline scripts read their inputs/outputs relative to environment
variables so the pipeline can be pointed at any local data directory
without editing code. Nothing in this repository ships the underlying
advertiser-level data (see data/README.md) — every script assumes the
four raw CSVs already exist under AD_DATA_ROOT.

Environment variables (all optional; sensible defaults shown):
    AD_DATA_ROOT   Root folder containing the four raw panel CSVs and
                   where step1 writes df_analysis_master.csv.
                   Default: ./data/raw
    STEP2_OUT      Output folder for step2/step3 (treatment-definition
                   and DiD) artifacts. Default: {AD_DATA_ROOT}/step2_treatment_output
    MC_OUT         Output folder for the Monte Carlo simulation.
                   Default: ./mc_output
    REFRAME_OUT    Output folder for the pivoted "first campaign type"
                   study (Track B application paper). Default:
                   {AD_DATA_ROOT}/reframe_output
    PAPER_ASSETS_OUT  Output folder for the auto-generated figures/tables
                   consumed by the paper. Default: ./paper_assets

Required raw input files (place under AD_DATA_ROOT):
    customer_day_panel.csv                 daily account-level panel
    customer_day_campaign_type_panel.csv   daily account x campaign-type panel
    customer_level_attributes.csv          static account attributes

See data/README.md for the expected schema of each file.
================================================================================
"""
import os
from pathlib import Path

AD_DATA_ROOT = Path(os.environ.get("AD_DATA_ROOT", "./data/raw"))
STEP2_OUT = Path(os.environ.get("STEP2_OUT", str(AD_DATA_ROOT / "step2_treatment_output")))
MC_OUT = Path(os.environ.get("MC_OUT", "./mc_output"))
REFRAME_OUT = Path(os.environ.get("REFRAME_OUT", str(AD_DATA_ROOT / "reframe_output")))
PAPER_ASSETS_OUT = Path(os.environ.get("PAPER_ASSETS_OUT", "./paper_assets"))

PANEL_PATH = AD_DATA_ROOT / "customer_day_panel.csv"
CTP_PATH = AD_DATA_ROOT / "customer_day_campaign_type_panel.csv"
ATTRS_PATH = AD_DATA_ROOT / "customer_level_attributes.csv"
MASTER_PATH = AD_DATA_ROOT / "df_analysis_master.csv"

# Target campaign type used for the discovery narrative (Sections 2-5 of the
# paper). Chosen because it is where the "born-treated" phenomenon was first
# discovered; the generalization study (step2e) repeats the diagnostic for
# every detected campaign type.
TARGET_TYPE = int(os.environ.get("TARGET_TYPE", "6"))

for d in (STEP2_OUT, MC_OUT, REFRAME_OUT, PAPER_ASSETS_OUT):
    Path(d).mkdir(parents=True, exist_ok=True)


def detect_date_column(df, candidates=("date", "stat_date", "stat_dt", "dt",
                                        "ad_date", "report_date", "log_date")):
    """Return the name of the date column in `df`, trying common names first
    and falling back to the first column that parses as a date."""
    import pandas as pd
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        try:
            pd.to_datetime(df[c].dropna().iloc[:5])
            return c
        except (ValueError, TypeError):
            continue
    raise KeyError(f"Could not auto-detect a date column. Columns present: {list(df.columns)}")
