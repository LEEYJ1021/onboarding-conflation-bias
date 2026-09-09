"""
================================================================================
step5b_dro_worst_case_bound.py
DRO Worst-Case Bound — eps 그리드에 걸친 최악의 경우 효과 하한 (Table 11 / Figure 13)
================================================================================
step5a가 저장한 원자료(track2_raw_data_type{T}.csv)와 캘리브레이션 매니페스트
(dro_calibration_manifest.json)를 읽어, KL-DRO 앙상블 반경(eps)을
0.25×~8× 캘리브레이션 수준으로 스윕하며 각 outcome의 최악의 경우 효과 하한을
계산한다. Breakdown 지점(하한이 정확히 0이 되는 eps)과 부호조건부 플라시오
검정은 step5c에서 별도로 다룬다 — 이 스크립트는 grid 자체만 책임진다.

수학적 배경:
    KL(P||Q)<=eps 제약 하에서 E_P[Y]의 최악의 경우 값은 쌍대문제
        min_λ>0  λ*log( E_Q[exp(Y/λ)] ) + λ*eps   (max일 때는 부호 반전)
    로 계산된다(Hu & Hong 2013). born-treated 쪽은 최소(min), 대조군 쪽은
    최대(max)로 왜곡해 두 왜곡분포 하 평균의 차이를 최악의 경우 효과로 삼는다.

출력 (dro_output_v4/):
    track2_dro_worst_case_bounds.csv (Table 11)
    fig_dro_bound_vs_epsilon_{outcome}.png (Figure 13)
================================================================================
"""
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = ["Noto Sans CJK KR", "Noto Sans"]
mpl.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================================
# 0. 경로 및 설정
# ============================================================================
ROOT = Path(os.environ.get("AD_DATA_ROOT", "./master_dataset"))
STEP2_DIR = Path(os.environ.get("STEP2_OUT", str(ROOT / "step2_treatment_output")))
OUT_DIR = Path(os.environ.get("DRO_OUT", str(STEP2_DIR / "dro_output_v4")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TYPE = int(os.environ.get("TARGET_TYPE_APPLICATION", "6"))
OUTCOME_VARS = ["log_spend_safe", "n_campaign_types_active"]
EPS_GRID_MULTIPLIERS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

RAW_DATA_PATH = OUT_DIR / f"track2_raw_data_type{TARGET_TYPE}.csv"
CALIB_MANIFEST_PATH = OUT_DIR / "dro_calibration_manifest.json"

if not RAW_DATA_PATH.exists() or not CALIB_MANIFEST_PATH.exists():
    raise FileNotFoundError("step5a_dro_calibration.py를 먼저 실행하세요 "
                             f"({RAW_DATA_PATH}, {CALIB_MANIFEST_PATH} 필요).")

import json
raw_df = pd.read_csv(RAW_DATA_PATH)
with open(CALIB_MANIFEST_PATH, "r", encoding="utf-8") as f:
    calib = json.load(f)
kl_calibrated_main = float(calib["eps_calibrated_main"])
print(f"캘리브레이션 eps(step5a에서 승계) = {kl_calibrated_main:.4f}")


# ============================================================================
# A. KL-DRO 최악의 경우 평균 (쌍대문제)
# ============================================================================
def kl_worst_case_mean(y: np.ndarray, eps: float, direction: str):
    y = np.asarray(y, dtype=float)
    sign = 1.0 if direction == "max" else -1.0
    y_signed = sign * y
    y_centered = y_signed - y_signed.max()

    def dual_objective(lam):
        if lam <= 1e-8:
            return np.inf
        log_mgf = np.log(np.mean(np.exp(y_centered / lam))) + y_signed.max() / lam
        return lam * log_mgf + lam * eps

    res = minimize_scalar(dual_objective, bounds=(1e-6, 1e6), method="bounded", options={"xatol": 1e-8})
    return float(sign * res.fun), float(res.x)


def worst_case_diff(y_bt: np.ndarray, y_ctrl: np.ndarray, eps: float) -> float:
    bt_worst, _ = kl_worst_case_mean(y_bt, eps, direction="min")
    ctrl_worst, _ = kl_worst_case_mean(y_ctrl, eps, direction="max")
    return bt_worst - ctrl_worst


# ============================================================================
# B. eps 그리드 스윕
# ============================================================================
print("\n" + "=" * 78)
print("B. DRO 최악의 경우 하한 — eps 그리드 스윕")
print("=" * 78)

dro_rows = []
for ycol in OUTCOME_VARS:
    sub = raw_df.dropna(subset=[ycol])
    y_bt = sub.loc[sub["is_born_treated"] == 1, ycol].to_numpy()
    y_ctrl = sub.loc[sub["is_born_treated"] == 0, ycol].to_numpy()
    naive_diff = float(y_bt.mean() - y_ctrl.mean())
    pooled_var = float(np.var(np.concatenate([y_bt, y_ctrl]), ddof=1))

    print(f"\n--- outcome: {ycol} (n_bt={len(y_bt)}, n_ctrl={len(y_ctrl)}) ---")
    print(f"  무보정(naive) 차이 = {naive_diff:+.4f}, 풀링 분산 Var(Y) = {pooled_var:.4f}")

    for mult in EPS_GRID_MULTIPLIERS:
        eps = mult * kl_calibrated_main
        wc = worst_case_diff(y_bt, y_ctrl, eps)
        dro_rows.append({"outcome": ycol, "eps_multiplier": mult, "eps": eps,
                          "naive_diff": naive_diff, "worst_case_diff": wc,
                          "pooled_variance": pooled_var, "survives_at_this_eps": bool(wc > 0)})
        print(f"  eps={eps:6.3f} (×{mult:>4.2f}) → 최악의 경우 하한 = {wc:+.4f} "
              f"{'✓ 여전히 양(+)' if wc > 0 else '✗ 0 이하로 붕괴'}")

dro_df = pd.DataFrame(dro_rows)
dro_path = OUT_DIR / "track2_dro_worst_case_bounds.csv"
dro_df.to_csv(dro_path, index=False)
print(f"\nDRO 하한 결과 저장: {dro_path}")


# ============================================================================
# C. 시각화 (Figure 13)
# ============================================================================
for ycol in OUTCOME_VARS:
    sub_plot = dro_df[dro_df["outcome"] == ycol].sort_values("eps")
    if sub_plot.empty:
        continue
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(sub_plot["eps"], sub_plot["worst_case_diff"], marker="o", color="firebrick",
            label="DRO 최악의 경우 하한")
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.axhline(sub_plot["naive_diff"].iloc[0], color="steelblue", linestyle="--", linewidth=1,
               label="무보정(naive) 점추정")
    ax.axvline(kl_calibrated_main, color="darkgreen", linestyle="--", linewidth=1,
               label=f"캘리브레이션 eps={kl_calibrated_main:.3f}")
    ax.set_xlabel("KL 앙상블 반경 (eps)")
    ax.set_ylabel(f"Track2 효과 하한 ({ycol})")
    ax.set_title(f"DRO 최악의 경우 하한 vs 앙상블 반경 — {ycol}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"fig_dro_bound_vs_epsilon_{ycol}.png", dpi=150)
    plt.close()

print(f"시각화 저장 위치: {OUT_DIR}")
print("\n완료 — 다음 단계: step5c_dro_breakdown_and_placebo_v4.py "
      "(breakdown 지점 및 부호조건부 플라시보 검정)")
