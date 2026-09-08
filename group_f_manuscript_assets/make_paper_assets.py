"""
================================================================================
make_paper_assets.py
--------------------------------------------------------------------------------
목적:
    앞서 실행한 진단/시뮬레이션 파이프라인(step1_data_integrity_*,
    step2_1~step3g, §2-일반화/§2-강건성-보완, mc_onboarding_conflation_bias 등)의
    산출물(csv/json)만을 입력으로 삼아, "Onboarding Conflation Bias" 논문에
    바로 삽입할 수 있는 그림(figures)과 표(tables)를 한 번에 생성한다.

    - 모든 텍스트/라벨/캡션은 영어
    - 모든 그림은 흑백(grayscale)만 사용 — color 대신 hatch/linestyle/marker로
      집단을 구분 (컬러 인쇄가 아닌 저널의 흑백 인쇄 규정을 그대로 통과하도록)
    - 300dpi PNG + 벡터 PDF를 함께 저장 (조판/투고 양쪽 대응)
    - 원본 파이프라인 스크립트를 다시 실행하지 않는다 — 이미 저장된
      csv/json만 읽는다. 즉 이 스크립트를 돌리기 전에 앞서 제시된
      step1~step3g / §2-일반화 / §2-강건성-보완 / mc_onboarding_conflation_bias.py를
      먼저 실행해 두어야 한다.
    - 특정 산출물이 없으면(아직 안 돌렸거나 경로가 다르면) 해당 그림/표만
      건너뛰고 경고만 남긴 뒤 나머지는 계속 진행한다(전체가 죽지 않음).

출력 폴더 구조:
    paper_assets/
    ├── figures/   (fig01_*.png/.pdf ... fig11_*.png/.pdf)
    ├── tables/    (table01_*.csv/.tex ... table08_*.csv/.tex)
    ├── generation_log.json   (무엇이 생성됐고 무엇이 스킵됐는지 기록)
    └── README.md

사용법 (원본 파이프라인과 동일한 환경변수를 그대로 재사용):
    export AD_DATA_ROOT="/home/yjlee/Research/Ad_Training/master_dataset_진단시간축일관성통합"
    export STEP2_OUT="$AD_DATA_ROOT/step2_treatment_output"     # 기본값도 이와 동일
    export MC_OUT="./mc_output"                                  # MC 스크립트를 돌린 경로
    export REFRAME_OUT="$AD_DATA_ROOT/reframe_output"            # 필요 시(§ 응용논문용)
    export PAPER_ASSETS_OUT="./paper_assets"                     # 이번 산출물 저장 위치
    python make_paper_assets.py
================================================================================
"""
import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. 경로 설정 (기존 파이프라인 스크립트들과 동일한 환경변수 이름을 그대로 사용)
# ==============================================================================
AD_DATA_ROOT = Path(os.environ.get(
    "AD_DATA_ROOT",
    "/home/yjlee/Research/Ad_Training/master_dataset_진단시간축일관성통합"
))
STEP2_DIR = Path(os.environ.get("STEP2_OUT", str(AD_DATA_ROOT / "step2_treatment_output")))
ALL_TYPES_DIR = STEP2_DIR / "all_types"
MC_DIR = Path(os.environ.get("MC_OUT", "./mc_output"))
REFRAME_DIR = Path(os.environ.get("REFRAME_OUT", str(AD_DATA_ROOT / "reframe_output")))
TARGET_TYPE = int(os.environ.get("TARGET_TYPE", "6"))  # 발견 서사의 중심이 된 캠페인 유형

OUT_ROOT = Path(os.environ.get("PAPER_ASSETS_OUT", "./paper_assets"))
FIG_DIR = OUT_ROOT / "figures"
TAB_DIR = OUT_ROOT / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

GENERATION_LOG = []  # {"item": ..., "status": "ok"/"skipped"/"error", "detail": ...}


def log(item: str, status: str, detail: str = ""):
    GENERATION_LOG.append({"item": item, "status": status, "detail": detail})
    tag = {"ok": "[OK]   ", "skipped": "[SKIP] ", "error": "[ERROR]"}.get(status, "[?]    ")
    print(f"{tag} {item}" + (f" — {detail}" if detail else ""))


# ==============================================================================
# 1. 스타일 — 흑백 / 탑저널 가독성 기준
# ==============================================================================
GRAYS = ["#000000", "#4D4D4D", "#808080", "#B3B3B3", "#D9D9D9"]
HATCHES = ["", "///", "...", "xxx", "\\\\\\\\"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]


def set_pub_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.color": "#E5E5E5",
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "patch.edgecolor": "black",
        "patch.linewidth": 0.8,
    })


set_pub_style()


def save_fig(fig, name: str, item_label: str):
    png_path = FIG_DIR / f"{name}.png"
    pdf_path = FIG_DIR / f"{name}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    log(item_label, "ok", f"{png_path.name} / {pdf_path.name}")


def save_table(df: pd.DataFrame, name: str, item_label: str, caption: str = ""):
    csv_path = TAB_DIR / f"{name}.csv"
    tex_path = TAB_DIR / f"{name}.tex"
    df.to_csv(csv_path, index=False)
    try:
        tex = df.to_latex(index=False, escape=True, na_rep="--", float_format="%.3f")
    except Exception:
        tex = df.to_latex(index=False, na_rep="--")
    header = f"% {caption}\n" if caption else ""
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(header + tex)
    log(item_label, "ok", f"{csv_path.name} / {tex_path.name}")


def safe_read_csv(path: Path):
    if not Path(path).exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def safe_read_json(path: Path):
    if not Path(path).exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def sig_stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ==============================================================================
# FIGURE 1 — Sample selection funnel (Step 1: data integrity)
# ==============================================================================
def fig01_sample_funnel():
    item = "Figure 1: Sample selection funnel"
    manifest = safe_read_json(AD_DATA_ROOT / "reproducibility_manifest.json")
    if manifest is None:
        log(item, "skipped", "reproducibility_manifest.json not found")
        return
    sel = manifest.get("sample_selection", {})
    stages = [
        ("Registry-matched\naccounts", sel.get("n_registry_matched")),
        ("Continuous\nobservation block", sel.get("n_continuous_block")),
        ("Stable window\nexists", sel.get("n_stable_window_exists")),
        ("Non-test /\nnon-anomalous", sel.get("n_clean_pre_mindays")),
        (f"Min. {sel.get('min_stable_window_days_rule', '?')}-day\nwindow rule", sel.get("n_clean_customers_final")),
    ]
    stages = [(lbl, n) for lbl, n in stages if n is not None]
    if not stages:
        log(item, "skipped", "no stage counts in manifest")
        return

    labels = [s[0] for s in stages]
    counts = [s[1] for s in stages]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = np.arange(len(labels))[::-1]
    bars = ax.barh(y, counts, height=0.55, facecolor="white", edgecolor="black",
                    hatch=HATCHES[0], linewidth=1.1)
    # 단계 간 손실을 옅은 회색 삼각형/텍스트로 표시
    for i in range(len(counts) - 1):
        drop = counts[i] - counts[i + 1]
        ax.annotate(f"-{drop}", xy=(counts[i + 1], y[i] - 0.5), xytext=(counts[i + 1], y[i] - 0.5),
                    fontsize=8.5, color="#4D4D4D", ha="left", va="center", style="italic")
    for yi, c in zip(y, counts):
        ax.text(c + max(counts) * 0.015, yi, f"n = {c}", va="center", ha="left",
                fontsize=10, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Number of advertiser accounts")
    ax.set_title("Sample Selection Funnel (Naive Panel Construction)")
    ax.set_xlim(0, max(counts) * 1.18)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_fig(fig, "fig01_sample_selection_funnel", item)


# ==============================================================================
# FIGURE 2 — Final cohort classification (naive vs. recovered) for TARGET_TYPE
# ==============================================================================
def fig02_cohort_classification():
    item = f"Figure 2: Cohort classification (campaign type {TARGET_TYPE})"
    summary = safe_read_json(STEP2_DIR / f"staggered_adoption_FINAL_type{TARGET_TYPE}_summary.json")
    if summary is None:
        log(item, "skipped", f"staggered_adoption_FINAL_type{TARGET_TYPE}_summary.json not found")
        return
    counts = summary.get("final_cohort_counts", {})
    order = ["never_treated", "still_left_censored", "robustness_only_adopter", "primary_adopter"]
    nice = {
        "never_treated": "Never-treated",
        "still_left_censored": "Still left-censored\n(unrecoverable)",
        "robustness_only_adopter": "Recovered\n(CAUTION)",
        "primary_adopter": "Recovered\n(SAFE) / within-window",
    }
    labels = [nice[k] for k in order if k in counts]
    vals = [counts[k] for k in order if k in counts]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, width=0.55, facecolor="white", edgecolor="black", linewidth=1.1)
    for b, h in zip(bars, HATCHES[:len(bars)]):
        b.set_hatch(h)
    for xi, v in zip(x, vals):
        ax.text(xi, v + max(vals) * 0.02, str(v), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Number of advertisers")
    ax.set_title(f"Final Cohort Classification After Left-Censoring Recovery\n(Campaign Type {TARGET_TYPE})")
    ax.set_ylim(0, max(vals) * 1.25)
    fig.tight_layout()
    save_fig(fig, "fig02_cohort_classification", item)


# ==============================================================================
# FIGURE 3 (KEY) — Gap-day distribution: the born-treated discovery figure
# ==============================================================================
def fig03_gap_day_distribution():
    item = f"Figure 3: Gap-day distribution (type {TARGET_TYPE}) — born-treated discovery"
    diag = safe_read_csv(STEP2_DIR / f"step3c_born_treated_diagnosis_type{TARGET_TYPE}.csv")
    if diag is None or "gap_days" not in diag.columns:
        log(item, "skipped", f"step3c_born_treated_diagnosis_type{TARGET_TYPE}.csv not found")
        return

    gap = diag["gap_days"].dropna().values
    threshold = 1  # BORN_TREATED_GAP_THRESHOLD used in the pipeline

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    max_gap = int(gap.max())
    bins = np.arange(0, max_gap + 3, 2)
    ax.hist(gap, bins=bins, facecolor="#D9D9D9", edgecolor="black", linewidth=0.9)
    ax.axvline(threshold + 0.5, color="black", linestyle="--", linewidth=1.3)
    ax.text(threshold + 1.5, ax.get_ylim()[1] * 0.92,
            f"Adopted threshold\n(gap $\\leq$ {threshold} day)", fontsize=9, ha="left", va="top")

    n_born = int((diag["gap_days"] <= threshold).sum())
    n_switch = int((diag["gap_days"] > threshold).sum())
    ax.annotate("Born-treated\n(panel start = adoption date)",
                xy=(0.5, ax.get_ylim()[1] * 0.55), xytext=(max_gap * 0.28, ax.get_ylim()[1] * 0.75),
                fontsize=9.5, ha="center",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
    ax.annotate("True switchers\n(pre-treatment history exists)",
                xy=(max_gap * 0.75, 1.0), xytext=(max_gap * 0.62, ax.get_ylim()[1] * 0.55),
                fontsize=9.5, ha="center",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0))

    ax.set_xlabel("Gap between panel-entry date and first adoption date (days)")
    ax.set_ylabel("Number of adopters")
    ax.set_title(f"Distribution of Adoption Gap Days (Campaign Type {TARGET_TYPE})\n"
                 f"Born-treated: n = {n_born}   |   True switchers: n = {n_switch}")
    fig.tight_layout()
    save_fig(fig, "fig03_gap_day_distribution", item)


# ==============================================================================
# FIGURE 4 — Cross-type generalization: born-treated ratio by campaign type
# ==============================================================================
def fig04_born_treated_ratio_by_type():
    item = "Figure 4: Born-treated ratio across campaign types"
    df = safe_read_csv(ALL_TYPES_DIR / "cohort_summary_all_types.csv")
    if df is None:
        log(item, "skipped", "all_types/cohort_summary_all_types.csv not found")
        return
    df = df.dropna(subset=["born_treated_ratio"]).sort_values("campaign_type")

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    x = np.arange(len(df))
    bars = ax.bar(x, df["born_treated_ratio"].values, width=0.55,
                   facecolor="white", edgecolor="black", linewidth=1.1)
    for i, b in enumerate(bars):
        b.set_hatch(HATCHES[i % len(HATCHES)])
    for xi, (r, n_tot, n_born) in enumerate(zip(df["born_treated_ratio"], df["n_final_adopters_total"], df["n_born_treated"])):
        ax.text(xi, r + 0.02, f"{r:.0%}\n(n={int(n_born)}/{int(n_tot)})", ha="center", va="bottom", fontsize=8.7)
    ax.axhline(df["born_treated_ratio"].mean(), color="black", linestyle=":", linewidth=1.0)
    ax.text(len(df) - 0.35, df["born_treated_ratio"].mean() + 0.015,
            f"Mean = {df['born_treated_ratio'].mean():.1%}", fontsize=8.5, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Type {int(t)}" for t in df["campaign_type"]])
    ax.set_ylabel("Born-treated ratio")
    ax.set_ylim(0, 1.15)
    ax.set_title("Born-Treated Ratio Is Homogeneous Across Campaign Types\n"
                 "(structural pattern, not a type-6-specific artifact)")
    fig.tight_layout()
    save_fig(fig, "fig04_born_treated_ratio_by_type", item)


# ==============================================================================
# FIGURE 5 — Gap-threshold sensitivity, by campaign type
# ==============================================================================
def fig05_gap_threshold_sensitivity():
    item = "Figure 5: Gap-threshold sensitivity by campaign type"
    df = safe_read_csv(ALL_TYPES_DIR / "gap_threshold_sensitivity_summary.csv")
    if df is None:
        log(item, "skipped", "all_types/gap_threshold_sensitivity_summary.csv not found")
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    types = sorted(df["campaign_type"].unique())
    for i, t in enumerate(types):
        sub = df[df["campaign_type"] == t].sort_values("gap_threshold_days")
        ax.plot(sub["gap_threshold_days"], sub["born_treated_ratio"],
                marker=MARKERS[i % len(MARKERS)], color="black",
                linestyle=LINESTYLES[i % len(LINESTYLES)], markerfacecolor="white",
                markeredgecolor="black", linewidth=1.3, markersize=7,
                label=f"Type {int(t)}")
    ax.axvline(1, color="#B3B3B3", linestyle="-", linewidth=6, alpha=0.5, zorder=0)
    ax.text(1, 0.05, "adopted\nthreshold", fontsize=8, ha="center", va="bottom", color="#4D4D4D")
    ax.set_xlabel("Gap-day threshold (days)")
    ax.set_ylabel("Born-treated ratio")
    ax.set_ylim(0, 1.05)
    ax.set_title("Robustness of the Born-Treated Classification to Threshold Choice")
    ax.legend(ncol=len(types), loc="lower center", bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    save_fig(fig, "fig05_gap_threshold_sensitivity", item)


# ==============================================================================
# FIGURE 6 — Monte Carlo: bias vs. p_born (Track 1 vs. Track 2)
# ==============================================================================
def fig06_mc_bias_vs_pborn():
    item = "Figure 6: Monte Carlo bias vs. born-treated share"
    df = safe_read_csv(MC_DIR / "mc_summary_by_p_born.csv")
    if df is None:
        log(item, "skipped", "mc_output/mc_summary_by_p_born.csv not found")
        return
    df = df.sort_values("p_born")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.axvspan(0.70, 0.85, color="#E5E5E5", zorder=0, label="Empirically observed range (74-81%)")
    ax.plot(df["p_born"], df["track1_bias"], marker="o", color="black", linestyle="-",
            markerfacecolor="white", markeredgecolor="black", linewidth=1.4, markersize=6,
            label="Track 1 (within-customer DiD)")
    ax.plot(df["p_born"], df["track2_bias"], marker="s", color="black", linestyle="--",
            markerfacecolor="black", markeredgecolor="black", linewidth=1.4, markersize=6,
            label="Track 2 (cross-sectional matching)")
    ax.set_xlabel("Simulated born-treated share ($p_{born}$)")
    ax.set_ylabel("Bias (estimate $-$ true effect)")
    ax.set_title("Monte Carlo Validation: Track 1 Remains Unbiased,\nTrack 2 Inherits a Novelty-Effect Bias")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_fig(fig, "fig06_mc_bias_vs_pborn", item)


# ==============================================================================
# FIGURE 7 — Monte Carlo: nominal N vs. effective N (Track 1 power collapse)
# ==============================================================================
def fig07_mc_effective_n():
    item = "Figure 7: Monte Carlo effective vs. nominal N"
    df = safe_read_csv(MC_DIR / "mc_summary_by_p_born.csv")
    if df is None:
        log(item, "skipped", "mc_output/mc_summary_by_p_born.csv not found")
        return
    df = df.sort_values("p_born")
    n_nominal = df["n_nominal"].iloc[0]

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.axvspan(0.70, 0.85, color="#E5E5E5", zorder=0)
    ax.plot(df["p_born"], df["avg_n_switch"], marker="o", color="black", linestyle="-",
            markerfacecolor="white", markeredgecolor="black", linewidth=1.4, markersize=6,
            label="Effective $N$ (identifiable switchers)")
    ax.axhline(n_nominal, color="black", linestyle="--", linewidth=1.2,
               label=f"Nominal $N$ = {int(n_nominal)} (naively assumed)")
    ax.set_xlabel("Simulated born-treated share ($p_{born}$)")
    ax.set_ylabel("Number of adopters")
    ax.set_title("Nominal Adopter Count Overstates the\nEffective Identifying Sample as Born-Treated Share Rises")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_fig(fig, "fig07_mc_effective_n", item)


# ==============================================================================
# FIGURE 8 — Monte Carlo: SE understatement (overconfidence)
# ==============================================================================
def fig08_mc_se_overconfidence():
    item = "Figure 8: Monte Carlo standard-error understatement"
    df = safe_read_csv(MC_DIR / "mc_summary_by_p_born.csv")
    if df is None:
        log(item, "skipped", "mc_output/mc_summary_by_p_born.csv not found")
        return
    df = df.sort_values("p_born")

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.axvspan(0.70, 0.85, color="#E5E5E5", zorder=0)
    ax.plot(df["p_born"], df["track1_se_correct_avg"], marker="o", color="black", linestyle="-",
            markerfacecolor="white", markeredgecolor="black", linewidth=1.4, markersize=6,
            label="Correct SE (uses effective $N$)")
    ax.plot(df["p_born"], df["track1_se_naive_nominal_avg"], marker="s", color="black", linestyle="--",
            markerfacecolor="black", markeredgecolor="black", linewidth=1.4, markersize=6,
            label="Naive SE (uses nominal $N$)")
    ax.set_xlabel("Simulated born-treated share ($p_{born}$)")
    ax.set_ylabel("Standard error of the pooled ATT")
    ax.set_title("Using the Nominal Sample Size Understates the\nStandard Error and Overstates Confidence")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_fig(fig, "fig08_mc_se_overconfidence", item)


# ==============================================================================
# FIGURE 9 — Track 1 case-level ATT + leave-one-out instability (forest-style)
# ==============================================================================
def fig09_track1_forest_loo():
    item = "Figure 9: Track 1 case-level ATT and leave-one-out sensitivity"
    cases = safe_read_csv(STEP2_DIR / f"step3d_track1_switcher_cases_type{TARGET_TYPE}.csv")
    loo = safe_read_csv(STEP2_DIR / f"step3e_track1_leave_one_out_type{TARGET_TYPE}.csv")
    if cases is None:
        log(item, "skipped", f"step3d_track1_switcher_cases_type{TARGET_TYPE}.csv not found")
        return

    outcomes = [o for o in cases["outcome"].unique()]
    fig, axes = plt.subplots(1, len(outcomes), figsize=(6.6 * len(outcomes), 4.4), squeeze=False)
    for j, oc in enumerate(outcomes):
        ax = axes[0][j]
        sub = cases[cases["outcome"] == oc].sort_values("att_i").reset_index(drop=True)
        y = np.arange(len(sub))
        ax.errorbar(sub["att_i"], y, xerr=0, fmt="o", color="black", markerfacecolor="white",
                    markeredgecolor="black", markersize=7, capsize=0)
        mean_att = sub["att_i"].mean()
        ax.axvline(mean_att, color="black", linestyle="--", linewidth=1.2,
                   label=f"Pooled mean = {mean_att:+.3f}")
        ax.axvline(0, color="#808080", linestyle=":", linewidth=1.0)

        # 가능하면 leave-one-out shift를 각 점 옆에 함께 표기 (n<=7 소표본의
        # 불안정성을 forest plot 위에서 바로 보여주기 위함)
        if loo is not None and "dropped_customer_id" in loo.columns:
            loo_oc = loo[loo["outcome"] == oc].set_index("dropped_customer_id")
            for yi, row in zip(y, sub.itertuples()):
                if row.customer_id in loo_oc.index:
                    shift = loo_oc.loc[row.customer_id, "shift_from_full_mean"]
                    ax.text(row.att_i, yi + 0.18, f"LOO $\\Delta$={shift:+.2f}",
                            fontsize=7, ha="center", va="bottom", color="#4D4D4D")

        ax.set_yticks(y)
        ax.set_yticklabels([f"ID {int(c)}" for c in sub["customer_id"]], fontsize=9)
        ax.set_xlabel(f"Individual ATT$_i$: {oc}")
        ax.set_title(f"n = {len(sub)}")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Track 1 (True Switchers): Case-Level Treatment Effects Are Small-N and Sign-Mixed",
                 fontsize=12, fontweight="bold", y=1.04)
    fig.tight_layout()
    save_fig(fig, "fig09_track1_case_level_att", item)


# ==============================================================================
# FIGURE 10 — Track 2: born-treated vs. matched control, early outcomes
# ==============================================================================
def fig10_track2_comparison():
    item = "Figure 10: Track 2 born-treated vs. matched-control comparison"
    df = safe_read_csv(STEP2_DIR / f"step3d_track2_born_treated_comparison_type{TARGET_TYPE}.csv")
    if df is None:
        log(item, "skipped", f"step3d_track2_born_treated_comparison_type{TARGET_TYPE}.csv not found")
        return

    outcome_pairs = [
        ("log_spend_safe_born_treated", "log_spend_safe_control_mean", "Cumulative log-spend\n(first 30 days)"),
        ("n_campaign_types_active_born_treated", "n_campaign_types_active_control_mean",
         "Avg. active campaign\ntypes (first 30 days)"),
    ]
    outcome_pairs = [(a, b, lbl) for a, b, lbl in outcome_pairs if a in df.columns and b in df.columns]
    if not outcome_pairs:
        log(item, "skipped", "expected columns not found in comparison csv")
        return

    fig, axes = plt.subplots(1, len(outcome_pairs), figsize=(5.6 * len(outcome_pairs), 4.4), squeeze=False)
    for j, (col_bt, col_ctrl, label) in enumerate(outcome_pairs):
        ax = axes[0][j]
        bt_vals = df[col_bt].dropna()
        ctrl_vals = df[col_ctrl].dropna()
        means = [bt_vals.mean(), ctrl_vals.mean()]
        sems = [bt_vals.std(ddof=1) / np.sqrt(len(bt_vals)), ctrl_vals.std(ddof=1) / np.sqrt(len(ctrl_vals))]
        x = np.arange(2)
        bars = ax.bar(x, means, yerr=sems, width=0.5, capsize=5,
                       facecolor="white", edgecolor="black", linewidth=1.1,
                       error_kw=dict(ecolor="black", elinewidth=1.2))
        bars[0].set_hatch("///")
        bars[1].set_hatch("")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Born-treated\n(n={len(bt_vals)})", f"Matched control\n(n={len(ctrl_vals)})"])
        ax.set_ylabel(label)
    fig.suptitle("Track 2: Born-Treated Advertisers Show Larger Early-Stage Outcomes\n"
                 "(descriptive association, not a causal estimate)",
                 fontsize=12, fontweight="bold", y=1.05)
    fig.tight_layout()
    save_fig(fig, "fig10_track2_comparison", item)


# ==============================================================================
# FIGURE 11 — Diagnostic protocol flowchart (schematic box-and-arrow diagram)
# ==============================================================================
def fig11_protocol_flowchart():
    item = "Figure 11: Diagnostic protocol flowchart"
    steps = [
        "A. Construct stable-observation-window sample\n(registry match -> continuous block -> stability filter)",
        "B. Classify adoption cohorts within the window\n(never-treated / adopted-within-window / left-censored)",
        "C. Recompute true first-active date from raw panel\n(look back beyond the stability filter)",
        "D. Pre-window safety check\n(anomaly flags, continuity, volatility, extreme-value density)",
        "E. Gap-day diagnostic\n(panel-entry date vs. recovered adoption date)",
    ]
    branches = [
        "Track 1\nWithin-customer DiD\n(true switchers only)",
        "Track 2\nCross-sectional matching\n(born-treated advertisers)",
    ]

    fig, ax = plt.subplots(figsize=(7.6, 9.6))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)

    box_w, box_h = 8.4, 1.55
    x0 = (10 - box_w) / 2
    ys = [12.0, 10.0, 8.0, 6.0, 4.0]
    for i, (y, text) in enumerate(zip(ys, steps)):
        box = FancyBboxPatch((x0, y - box_h / 2), box_w, box_h,
                              boxstyle="round,pad=0.12,rounding_size=0.12",
                              facecolor="white", edgecolor="black", linewidth=1.3)
        ax.add_patch(box)
        ax.text(x0 + box_w / 2, y, text, ha="center", va="center", fontsize=9.3, linespacing=1.4)
        if i < len(ys) - 1:
            ax.annotate("", xy=(x0 + box_w / 2, ys[i + 1] + box_h / 2 + 0.05),
                        xytext=(x0 + box_w / 2, y - box_h / 2 - 0.05),
                        arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.3))

    # 마지막 단계(E)에서 두 갈래로 분기
    branch_y = 1.6
    branch_w, branch_h = 3.6, 1.5
    branch_xs = [10 / 2 - branch_w - 0.5, 10 / 2 + 0.5]
    for bx, label in zip(branch_xs, branches):
        box = FancyBboxPatch((bx, branch_y - branch_h / 2), branch_w, branch_h,
                              boxstyle="round,pad=0.12,rounding_size=0.12",
                              facecolor="#F2F2F2", edgecolor="black", linewidth=1.3)
        ax.add_patch(box)
        ax.text(bx + branch_w / 2, branch_y, label, ha="center", va="center",
                fontsize=9.5, fontweight="bold", linespacing=1.5)
        ax.annotate("", xy=(bx + branch_w / 2, branch_y + branch_h / 2 + 0.05),
                    xytext=(x0 + box_w / 2, ys[-1] - box_h / 2 - 0.05),
                    arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.2,
                                     connectionstyle="arc3,rad=0.0" if bx < 5 else "arc3,rad=0.0"))

    ax.text(branch_xs[0] + branch_w / 2, branch_y - branch_h / 2 - 0.35,
            "gap $>$ threshold", ha="center", fontsize=8.3, style="italic")
    ax.text(branch_xs[1] + branch_w / 2, branch_y - branch_h / 2 - 0.35,
            "gap $\\leq$ threshold", ha="center", fontsize=8.3, style="italic")

    ax.set_title("The Gap-Day Diagnostic Protocol", fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    save_fig(fig, "fig11_diagnostic_protocol_flowchart", item)


# ==============================================================================
# TABLE 1 — Sample selection stages
# ==============================================================================
def table01_sample_selection():
    item = "Table 1: Sample selection stages"
    manifest = safe_read_json(AD_DATA_ROOT / "reproducibility_manifest.json")
    if manifest is None:
        log(item, "skipped", "reproducibility_manifest.json not found")
        return
    sel = manifest.get("sample_selection", {})
    rows = [
        ("Registry-matched accounts", sel.get("n_registry_matched")),
        ("Continuous observation block", sel.get("n_continuous_block")),
        ("Stable window exists", sel.get("n_stable_window_exists")),
        ("Non-test / non-billing-anomalous", sel.get("n_clean_pre_mindays")),
        (f"Minimum {sel.get('min_stable_window_days_rule', '?')}-day window rule",
         sel.get("n_clean_customers_final")),
    ]
    df = pd.DataFrame(rows, columns=["Selection stage", "N remaining"])
    df["N excluded at this stage"] = [np.nan] + list(-np.diff(df["N remaining"].values))
    save_table(df, "table01_sample_selection_stages", item,
               caption="Sample selection funnel for the naive stable-window panel.")


# ==============================================================================
# TABLE 2 — Final cohort classification
# ==============================================================================
def table02_cohort_classification():
    item = f"Table 2: Cohort classification (type {TARGET_TYPE})"
    summary = safe_read_json(STEP2_DIR / f"staggered_adoption_FINAL_type{TARGET_TYPE}_summary.json")
    if summary is None:
        log(item, "skipped", f"staggered_adoption_FINAL_type{TARGET_TYPE}_summary.json not found")
        return
    counts = summary.get("final_cohort_counts", {})
    nice = {
        "never_treated": "Never-treated",
        "still_left_censored": "Still left-censored (unrecoverable)",
        "robustness_only_adopter": "Recovered, CAUTION-flagged",
        "primary_adopter": "Adopted within window / recovered SAFE",
    }
    df = pd.DataFrame({"Final cohort": [nice.get(k, k) for k in counts],
                        "N": list(counts.values())})
    save_table(df, "table02_final_cohort_classification", item,
               caption=f"Final cohort classification after left-censoring recovery (campaign type {TARGET_TYPE}).")


# ==============================================================================
# TABLE 3 — Cross-type generalization summary
# ==============================================================================
def table03_cross_type_summary():
    item = "Table 3: Cross-type generalization summary"
    df = safe_read_csv(ALL_TYPES_DIR / "cohort_summary_all_types.csv")
    if df is None:
        log(item, "skipped", "all_types/cohort_summary_all_types.csv not found")
        return
    keep = ["campaign_type", "n_ever_active", "n_final_adopters_total",
            "n_born_treated", "n_true_switcher", "born_treated_ratio", "n_valid_stack_for_DiD"]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    rename = {
        "campaign_type": "Campaign type", "n_ever_active": "Ever active",
        "n_final_adopters_total": "Final adopters", "n_born_treated": "Born-treated",
        "n_true_switcher": "True switchers", "born_treated_ratio": "Born-treated ratio",
        "n_valid_stack_for_DiD": "Valid DiD stacks",
    }
    out = out.rename(columns=rename)
    save_table(out, "table03_cross_type_summary", item,
               caption="Born-treated diagnostic results replicated across all observed campaign types.")


# ==============================================================================
# TABLE 4 — Homogeneity tests: chi-square vs. Monte Carlo exact test
# ==============================================================================
def table04_homogeneity_tests():
    item = "Table 4: Homogeneity tests (chi-square vs. MC exact)"
    df = safe_read_csv(ALL_TYPES_DIR / "homogeneity_chisq_vs_mc_exact.csv")
    if df is None:
        log(item, "skipped", "all_types/homogeneity_chisq_vs_mc_exact.csv not found")
        return
    keep = ["gap_threshold_days", "chi2_observed", "min_expected_cell",
            "asymptotic_pvalue", "mc_exact_pvalue", "agree_at_alpha_05"]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    rename = {
        "gap_threshold_days": "Gap threshold (days)", "chi2_observed": "Chi-square statistic",
        "min_expected_cell": "Min. expected cell count", "asymptotic_pvalue": "Asymptotic $p$",
        "mc_exact_pvalue": "MC exact $p$", "agree_at_alpha_05": "Agree at $\\alpha=.05$",
    }
    out = out.rename(columns=rename)
    save_table(out, "table04_homogeneity_tests", item,
               caption="Cross-type homogeneity of the born-treated ratio: asymptotic vs. permutation-exact tests.")


# ==============================================================================
# TABLE 5 — Gap-threshold sensitivity (pivoted, type x threshold)
# ==============================================================================
def table05_gap_threshold_sensitivity():
    item = "Table 5: Gap-threshold sensitivity (pivoted)"
    df = safe_read_csv(ALL_TYPES_DIR / "gap_threshold_sensitivity_summary.csv")
    if df is None:
        log(item, "skipped", "all_types/gap_threshold_sensitivity_summary.csv not found")
        return
    pivot = df.pivot(index="campaign_type", columns="gap_threshold_days", values="born_treated_ratio")
    pivot.columns = [f"threshold = {int(c)}d" for c in pivot.columns]
    pivot = pivot.reset_index().rename(columns={"campaign_type": "Campaign type"})
    save_table(pivot, "table05_gap_threshold_sensitivity", item,
               caption="Born-treated ratio by campaign type across alternative gap-day thresholds.")


# ==============================================================================
# TABLE 6 — Monte Carlo summary by p_born
# ==============================================================================
def table06_mc_summary():
    item = "Table 6: Monte Carlo summary by born-treated share"
    df = safe_read_csv(MC_DIR / "mc_summary_by_p_born.csv")
    if df is None:
        log(item, "skipped", "mc_output/mc_summary_by_p_born.csv not found")
        return
    keep = ["p_born", "avg_n_switch", "track1_bias", "track1_se_correct_avg",
            "track1_se_naive_nominal_avg", "track1_se_understatement_pct",
            "track2_bias", "track2_pct_significant"]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy().sort_values("p_born")
    rename = {
        "p_born": "$p_{born}$", "avg_n_switch": "Effective $N$", "track1_bias": "Track 1 bias",
        "track1_se_correct_avg": "Track 1 SE (correct)", "track1_se_naive_nominal_avg": "Track 1 SE (naive)",
        "track1_se_understatement_pct": "SE understatement (%)", "track2_bias": "Track 2 bias",
        "track2_pct_significant": "Track 2 % significant",
    }
    out = out.rename(columns=rename)
    save_table(out, "table06_monte_carlo_summary", item,
               caption="Monte Carlo results across the simulated born-treated share grid.")


# ==============================================================================
# TABLE 7 — Track 1 case-level ATT
# ==============================================================================
def table07_track1_cases():
    item = "Table 7: Track 1 case-level ATT"
    df = safe_read_csv(STEP2_DIR / f"step3d_track1_switcher_cases_type{TARGET_TYPE}.csv")
    if df is None:
        log(item, "skipped", f"step3d_track1_switcher_cases_type{TARGET_TYPE}.csv not found")
        return
    keep = ["customer_id", "outcome", "n_pre_days", "n_post_days", "att_i"]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy().sort_values(["outcome", "att_i"])
    rename = {
        "customer_id": "Advertiser ID", "outcome": "Outcome", "n_pre_days": "Pre-period days",
        "n_post_days": "Post-period days", "att_i": "Individual ATT",
    }
    out = out.rename(columns=rename)
    save_table(out, "table07_track1_case_level_att", item,
               caption="Case-level treatment effects for true switchers (Track 1).")


# ==============================================================================
# TABLE 8 — Track 2 regression comparison (unadjusted / contaminated / clean covariate)
# ==============================================================================
def table08_track2_regression_comparison():
    item = "Table 8: Track 2 regression comparison"
    data = safe_read_json(STEP2_DIR / f"step3g_track2_regression_comparison_type{TARGET_TYPE}.json")
    if data is None:
        log(item, "skipped", f"step3g_track2_regression_comparison_type{TARGET_TYPE}.json not found")
        return
    rows = []
    for outcome, res in data.get("results", {}).items():
        for spec_key, spec_label in [
            ("unadjusted", "Unadjusted"),
            ("orig_contaminated_covariate", "Adjusted (structural covariate)"),
            ("clean_covariate", "Adjusted (outcome-window-excluded covariate)"),
        ]:
            spec = res.get(spec_key)
            if spec is None:
                continue
            rows.append({
                "Outcome": outcome, "Specification": spec_label,
                "Coefficient": spec.get("coef"), "p-value": spec.get("pvalue"),
                "Sig.": sig_stars(spec.get("pvalue")),
                "Shrinkage vs. unadjusted (%)": (
                    None if spec_key == "unadjusted" else
                    (spec.get("shrinkage_pct") * 100 if spec.get("shrinkage_pct") is not None else None)
                ),
            })
    if not rows:
        log(item, "skipped", "no results found in json")
        return
    out = pd.DataFrame(rows)
    save_table(out, "table08_track2_regression_comparison", item,
               caption="Track 2 covariate-adjustment comparison, including an outcome-contamination check.")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 78)
    print("Generating grayscale, publication-style figures and tables")
    print(f"  AD_DATA_ROOT = {AD_DATA_ROOT}")
    print(f"  STEP2_OUT    = {STEP2_DIR}")
    print(f"  MC_OUT       = {MC_DIR}")
    print(f"  TARGET_TYPE  = {TARGET_TYPE}")
    print(f"  Output       = {OUT_ROOT}")
    print("=" * 78)

    for fn in [
        fig01_sample_funnel, fig02_cohort_classification, fig03_gap_day_distribution,
        fig04_born_treated_ratio_by_type, fig05_gap_threshold_sensitivity,
        fig06_mc_bias_vs_pborn, fig07_mc_effective_n, fig08_mc_se_overconfidence,
        fig09_track1_forest_loo, fig10_track2_comparison, fig11_protocol_flowchart,
    ]:
        try:
            fn()
        except Exception as e:
            log(fn.__name__, "error", str(e))

    for fn in [
        table01_sample_selection, table02_cohort_classification, table03_cross_type_summary,
        table04_homogeneity_tests, table05_gap_threshold_sensitivity, table06_mc_summary,
        table07_track1_cases, table08_track2_regression_comparison,
    ]:
        try:
            fn()
        except Exception as e:
            log(fn.__name__, "error", str(e))

    with open(OUT_ROOT / "generation_log.json", "w", encoding="utf-8") as f:
        json.dump(GENERATION_LOG, f, ensure_ascii=False, indent=2)

    n_ok = sum(1 for r in GENERATION_LOG if r["status"] == "ok")
    n_skip = sum(1 for r in GENERATION_LOG if r["status"] == "skipped")
    n_err = sum(1 for r in GENERATION_LOG if r["status"] == "error")
    print("\n" + "=" * 78)
    print(f"Done: {n_ok} generated, {n_skip} skipped (missing inputs), {n_err} errors")
    print(f"Figures -> {FIG_DIR}")
    print(f"Tables  -> {TAB_DIR}")
    print(f"Log     -> {OUT_ROOT / 'generation_log.json'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
