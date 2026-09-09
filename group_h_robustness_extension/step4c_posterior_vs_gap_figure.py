"""
================================================================================
step4c_posterior_vs_gap_figure.py
Figure 12B — 안정화(ridge) 혼합모형의 전체 N=151 사후확률 산점도 (버블=고객수)
================================================================================
목적:
    step4b가 산출한 hard_vs_soft_classification_v2.csv를 그대로 다시 읽어
    그림만 생성하는 순수 포맷팅 스크립트다(§20 "Group F: 절대 분석을 재실행하지
    않는다"는 원칙을 Group H 산출물에도 동일하게 적용). N=151명 중 118명이
    gap_days=0을 공유하는 등 동일한 (gap_days, posterior) 조합에 여러 고객이
    겹치므로, 단순 산점도는 겹침(overplotting)으로 실제 밀집도를 왜곡한다.
    따라서 동일 좌표를 공유하는 고객 수를 세어 버블 크기로 인코딩한다.

    이 그림은 "애매 구간[0.2,0.8]에 5/151명만 존재하고 대부분의 질량이
    양 극단에 쏠려 있다"는 §23.1의 핵심 시각적 근거다(README Figure 12B).

입력: censored_mixture_output_v2/hard_vs_soft_classification_v2.csv (step4b 출력)
출력: censored_mixture_output_v2/fig_posterior_vs_gap_v2_bubble.png
================================================================================
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = ["Noto Sans CJK KR", "Noto Sans"]
mpl.rcParams["axes.unicode_minus"] = False

# ============================================================================
# 0. 경로 및 설정
# ============================================================================
ROOT = Path(os.environ.get("AD_DATA_ROOT", "./master_dataset"))
STEP2_DIR = Path(os.environ.get("STEP2_OUT", str(ROOT / "step2_treatment_output")))
MIXTURE_V2_DIR = Path(os.environ.get("CENSORED_OUT", str(STEP2_DIR / "censored_mixture_output_v2")))

AMBIGUOUS_LO = float(os.environ.get("AMBIGUOUS_LO", "0.2"))
AMBIGUOUS_HI = float(os.environ.get("AMBIGUOUS_HI", "0.8"))
POSTERIOR_ROUND_DECIMALS = int(os.environ.get("POSTERIOR_ROUND_DECIMALS", "3"))
INIT_THRESHOLD = int(os.environ.get("INIT_HARD_THRESHOLD_DAYS", "1"))

IN_PATH = MIXTURE_V2_DIR / "hard_vs_soft_classification_v2.csv"
if not IN_PATH.exists():
    raise FileNotFoundError(f"{IN_PATH} 없음 — step4b_mixture_v2_stabilized.py를 먼저 실행하세요.")

# ============================================================================
# A. 데이터 로드 및 버블 집계
# ============================================================================
print("=" * 78)
print("Figure 12B — 전체 N 사후확률 버블 산점도")
print("=" * 78)

df = pd.read_csv(IN_PATH)
n_total = len(df)
print(f"입력 고객 수: {n_total}명")

# 겹침 집계를 위해 사후확률을 소수 N자리로 반올림해 (gap_days, campaign_type,
# rounded_posterior) 조합별 고객 수를 센다. campaign_type을 키에 포함시키는
# 이유: 같은 gap_days·posterior라도 캠페인 유형이 다르면 별개 점으로 취급해야
# 색상 구분이 의미를 갖는다.
df["_posterior_rounded"] = df["posterior_born_treated_prob"].round(POSTERIOR_ROUND_DECIMALS)
bubble = (
    df.groupby(["gap_days", "campaign_type", "_posterior_rounded"])
    .size().rename("n_customers").reset_index()
)
bubble["posterior_born_treated_prob"] = bubble["_posterior_rounded"]

n_ambig = int(df["posterior_born_treated_prob"].between(AMBIGUOUS_LO, AMBIGUOUS_HI).sum())
print(f"애매 구간[{AMBIGUOUS_LO},{AMBIGUOUS_HI}] 고객 수: {n_ambig}/{n_total}명")
print(f"버블(고유 좌표) 개수: {len(bubble)}개 (겹침 압축 전 {n_total}개 점)")
print(bubble.sort_values("n_customers", ascending=False).head(5).to_string(index=False))

# ============================================================================
# B. 시각화
# ============================================================================
fig, ax = plt.subplots(figsize=(9, 5.5))

campaign_types = sorted(bubble["campaign_type"].unique())
color_map = {t: c for t, c in zip(campaign_types, plt.cm.tab10.colors)}
colors = bubble["campaign_type"].map(color_map)

# 버블 면적 ∝ 고객 수 (지름이 아니라 면적으로 인코딩 — 과장 방지)
MIN_SIZE, MAX_SIZE = 30, 900
n_max = bubble["n_customers"].max()
sizes = MIN_SIZE + (MAX_SIZE - MIN_SIZE) * (bubble["n_customers"] / n_max)

ax.axvspan(-0.5, INIT_THRESHOLD + 0.5, color="firebrick", alpha=0.04)
ax.axhspan(AMBIGUOUS_LO, AMBIGUOUS_HI, color="gray", alpha=0.06)
ax.axvline(INIT_THRESHOLD + 0.5, color="firebrick", linestyle="--", linewidth=1,
           label=f"기존 하드 threshold(gap<={INIT_THRESHOLD})")
ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)

sc = ax.scatter(bubble["gap_days"], bubble["posterior_born_treated_prob"],
                 s=sizes, c=colors, alpha=0.65, edgecolor="white", linewidth=0.8)

for _, row in bubble.iterrows():
    if row["n_customers"] >= 10:
        ax.annotate(str(int(row["n_customers"])), (row["gap_days"], row["posterior_born_treated_prob"]),
                    ha="center", va="center", fontsize=8, fontweight="bold")

legend_handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[t],
                              markersize=9, label=f"캠페인유형 {t}") for t in campaign_types]
ax.legend(handles=legend_handles, fontsize=8, loc="center right", title="버블 크기 = 고객 수")

ax.set_xlabel("gap_days")
ax.set_ylabel("사후확률 P(born-treated | gap, X) — 안정화(릿지) 모형")
ax.set_title(f"Figure 12B: 전체 N={n_total} 사후확률 분포 (애매구간 {n_ambig}명)")
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()

out_path = MIXTURE_V2_DIR / "fig_posterior_vs_gap_v2_bubble.png"
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\n저장: {out_path}")
print("완료 — 다음 단계: step5a_dro_calibration.py")
