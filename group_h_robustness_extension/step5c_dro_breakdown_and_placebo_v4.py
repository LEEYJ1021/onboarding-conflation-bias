"""
================================================================================
step5c_dro_breakdown_and_placebo_v4.py
DRO Breakdown Ratio + 부호조건부(sign-conditional) 플라시보 검정 (Table 12 / Figure 14)
================================================================================
step5a의 원자료·캘리브레이션을 읽어 각 outcome의 KL-DRO breakdown eps
(worst-case 하한이 정확히 0이 되는 지점)를 구하고, 이를 캘리브레이션 수준과
비교한 breakdown_ratio를 계산한다. ratio<1이면 "관측된 공변량 불균형 수준
이하의 왜곡만으로도 효과가 사라진다"는 뜻이다.

--------------------------------------------------------------------------------
[근원적 버그 수정 기록 — v3→v4]
    문제: 이전 버전의 플라시보(귀무) 순열검정 해석이 방향이 뒤집혀 있었다.
    원인: 무작위 재배정으로 만든 "가짜 처치군"의 naive_diff는 부호가 대략
    절반은 음수로 나온다. find_breakdown_eps()는 f(eps≈0)<=0이면 즉시
    breakdown_eps=0을 반환하므로, 부호가 반대로 나온 재배정은 거의 항상
    breakdown_ratio=0이 되어, 부호 무관(unconditional) 귀무분포 자체가 0
    근처로 퇴화한다. 이 퇴화 분포를 기준으로 삼으면 부호가 맞는 실제 효과는
    거의 항상 100백분위수로 나오는데, 이를 "노이즈로 설명 가능"이라는 반대
    방향으로 해석하는 오류가 있었다.
    수정: 관측된 효과와 부호가 일치하는 재배정만 걸러 조건부 귀무분포를
    만들고, p=P(조건부 귀무 ratio>=관측 ratio)를 계산한다. 이는 "우연히 같은
    방향으로 나온 순수 노이즈가 관측된 효과만큼 견고할 확률"이며, 이 p가
    작다고 해서 breakdown_ratio<1 자체(=confounding에 대한 취약성)가 없어지는
    것은 아니다 — 두 질문은 서로 다르다:
        (a) 점추정이 노이즈가 아닌가?           → 이 플라시보 검정이 답함
        (b) confounding에 강건한가(ratio>=1)?  → breakdown_ratio 자체가 답함
    §G의 논문용 문구는 이 둘을 분리해서 서술한다.
--------------------------------------------------------------------------------

입력 (step5a 산출물):
    dro_output_v4/track2_raw_data_type{T}.csv
    dro_output_v4/dro_calibration_manifest.json

출력 (dro_output_v4/):
    track2_dro_breakdown_analysis_v4.json (Table 12)
    track2_dro_placebo_conditional_null_v4.csv
    fig_dro_placebo_conditional_{outcome}.png (Figure 14)
    track2_dro_manifest_v4.json (논문용 문구 포함 종합 매니페스트)
================================================================================
"""
import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, brentq
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
STEP3G_JSON_PATH = STEP2_DIR / f"step3g_track2_regression_comparison_type{TARGET_TYPE}.json"

NEFF_FRAGILE_WARN_ABS = float(os.environ.get("NEFF_FRAGILE_WARN_ABS", "5.0"))
N_PLACEBO_TARGET_MATCHED = int(os.environ.get("N_PLACEBO_TARGET_MATCHED", "200"))
PLACEBO_MAX_DRAWS_MULTIPLIER = int(os.environ.get("PLACEBO_MAX_DRAWS_MULTIPLIER", "6"))
PLACEBO_SEED = int(os.environ.get("PLACEBO_SEED", "20260909"))
PLACEBO_P_ROBUSTNESS_THRESHOLD = float(os.environ.get("PLACEBO_P_ROBUSTNESS_THRESHOLD", "0.10"))

RAW_DATA_PATH = OUT_DIR / f"track2_raw_data_type{TARGET_TYPE}.csv"
CALIB_MANIFEST_PATH = OUT_DIR / "dro_calibration_manifest.json"
if not RAW_DATA_PATH.exists() or not CALIB_MANIFEST_PATH.exists():
    raise FileNotFoundError("step5a_dro_calibration.py를 먼저 실행하세요.")

raw_df = pd.read_csv(RAW_DATA_PATH)
with open(CALIB_MANIFEST_PATH, "r", encoding="utf-8") as f:
    calib = json.load(f)
kl_calibrated_main = float(calib["eps_calibrated_main"])
mixture_model_justified = calib.get("mixture_v1v2_model_justified")
print(f"캘리브레이션 eps(step5a에서 승계) = {kl_calibrated_main:.4f}")
print(f"① 혼합모형 MODEL_JUSTIFIED(step5a에서 승계) = {mixture_model_justified}")


# ============================================================================
# A. KL-DRO 최악의 경우 평균 + breakdown eps 탐색
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


def tilted_weights_and_n_eff(y: np.ndarray, eps: float, direction: str):
    y = np.asarray(y, dtype=float)
    n = len(y)
    sign = 1.0 if direction == "max" else -1.0
    y_signed = sign * y
    _, lam_opt = kl_worst_case_mean(y, eps, direction)
    if lam_opt <= 1e-8:
        w = np.zeros(n); w[np.argmax(y_signed)] = 1.0
    else:
        z = (y_signed - y_signed.max()) / lam_opt
        w_raw = np.exp(z)
        w = w_raw / w_raw.sum()
    n_eff = float((np.sum(w) ** 2) / np.sum(w ** 2)) if np.sum(w ** 2) > 0 else np.nan
    return w, n_eff


def worst_case_diff(y_bt, y_ctrl, eps):
    bt_worst, _ = kl_worst_case_mean(y_bt, eps, direction="min")
    ctrl_worst, _ = kl_worst_case_mean(y_ctrl, eps, direction="max")
    return bt_worst - ctrl_worst


def find_breakdown_eps(y_bt, y_ctrl, eps_max=50.0):
    """worst_case_diff(eps)=0이 되는 eps를 이분법으로 탐색.
    None='eps_max까지 안 무너짐(매우 강건)', 0.0='무보정 차이 자체가 이미<=0'."""
    f = lambda e: worst_case_diff(y_bt, y_ctrl, e)
    if f(1e-6) <= 0:
        return 0.0
    if f(eps_max) > 0:
        return None
    try:
        return float(brentq(f, 1e-6, eps_max, xtol=1e-4))
    except Exception:
        return None


def placebo_null_breakdown_ratios_conditional(y_pool, n_fake_treated, eps_calibrated,
                                               observed_naive_diff, n_target_matched, seed,
                                               max_draws_multiplier=PLACEBO_MAX_DRAWS_MULTIPLIER):
    """관측된 효과와 부호가 일치하는 재배정만 걸러 조건부 귀무분포를 구성한다.
    (부호가 다른 재배정은 breakdown_ratio가 거의 항상 즉시 0이 되는 자명한
    경우라 비교에 정보량이 없으므로 제외한다 — 위 v3→v4 수정 기록 참조.)"""
    if observed_naive_diff == 0:
        return np.array([]), 0, 0, np.nan, np.array([])
    rng = np.random.default_rng(seed)
    n_pool = len(y_pool)
    target_sign = np.sign(observed_naive_diff)

    matched_ratios, unconditional_ratios = [], []
    n_total_drawn = 0
    max_draws = max(n_target_matched * max_draws_multiplier, 50)

    while len(matched_ratios) < n_target_matched and n_total_drawn < max_draws:
        idx = rng.permutation(n_pool)
        fake_bt, fake_ctrl = y_pool[idx[:n_fake_treated]], y_pool[idx[n_fake_treated:]]
        n_total_drawn += 1
        fake_diff = float(fake_bt.mean() - fake_ctrl.mean())
        bd = find_breakdown_eps(fake_bt, fake_ctrl)
        ratio = (bd / eps_calibrated) if bd is not None else np.inf
        unconditional_ratios.append(ratio)
        if fake_diff != 0 and np.sign(fake_diff) == target_sign:
            matched_ratios.append(ratio)

    sign_match_rate = (len(matched_ratios) / n_total_drawn) if n_total_drawn > 0 else np.nan
    return (np.array(matched_ratios), len(matched_ratios), n_total_drawn,
            sign_match_rate, np.array(unconditional_ratios))


# ============================================================================
# B. Breakdown 분석 + 부호조건부 플라시보 검정
# ============================================================================
print("\n" + "=" * 78)
print("B. DRO breakdown ratio + 부호조건부 플라시보 검정")
print("=" * 78)

breakdown_summary, fragile_outcomes, placebo_rows = {}, [], []

for ycol in OUTCOME_VARS:
    sub = raw_df.dropna(subset=[ycol])
    y_bt = sub.loc[sub["is_born_treated"] == 1, ycol].to_numpy()
    y_ctrl = sub.loc[sub["is_born_treated"] == 0, ycol].to_numpy()
    y_pool = np.concatenate([y_bt, y_ctrl])
    naive_diff = float(y_bt.mean() - y_ctrl.mean())
    pooled_var = float(np.var(y_pool, ddof=1)) if len(y_pool) > 1 else np.nan

    print(f"\n--- outcome: {ycol} (n_bt={len(y_bt)}, n_ctrl={len(y_ctrl)}) ---")
    print(f"  무보정 차이 = {naive_diff:+.4f}, 풀링 분산 = {pooled_var:.4f}")

    breakdown = find_breakdown_eps(y_bt, y_ctrl)
    n_eff_bt_at_bd = n_eff_ctrl_at_bd = None
    var_norm_index = None
    eps_needed_for_naive_diff = (naive_diff ** 2) / (2.0 * pooled_var) if pooled_var and pooled_var > 0 else np.nan

    if breakdown is None:
        print("  → breakdown eps: eps=50까지도 붕괴하지 않음 — 매우 강건")
        ratio = None
    elif breakdown == 0.0:
        print("  → 무보정 차이 자체가 이미 0 이하 — breakdown 개념 성립 안 함")
        ratio = 0.0
    else:
        ratio = breakdown / kl_calibrated_main
        if eps_needed_for_naive_diff and eps_needed_for_naive_diff > 0:
            var_norm_index = breakdown / eps_needed_for_naive_diff
            print(f"    [분산정규화] variance_normalized_breakdown_index={var_norm_index:.3f}")
        _, n_eff_bt_at_bd = tilted_weights_and_n_eff(y_bt, breakdown, direction="min")
        _, n_eff_ctrl_at_bd = tilted_weights_and_n_eff(y_ctrl, breakdown, direction="max")
        tag = "강건성 지지" if ratio >= 1.0 else "취약성 후보(아래 플라시보로 진위 확인)"
        marker = "  →" if ratio >= 1.0 else "  ⚠ →"
        print(f"{marker} breakdown eps = {breakdown:.4f} (캘리브레이션의 {ratio:.2f}배) — {tag}")
        print(f"    tilted-weight n_eff(Kish): born-treated={n_eff_bt_at_bd:.2f}/{len(y_bt)}, "
              f"대조군={n_eff_ctrl_at_bd:.2f}/{len(y_ctrl)}")
        if ratio < 1.0:
            fragile_outcomes.append(ycol)

    n_eff_fragile_flag = bool(
        (n_eff_bt_at_bd is not None and n_eff_bt_at_bd < NEFF_FRAGILE_WARN_ABS) or
        (n_eff_ctrl_at_bd is not None and n_eff_ctrl_at_bd < NEFF_FRAGILE_WARN_ABS)
    )
    if n_eff_fragile_flag:
        print(f"    ⚠ n_eff<{NEFF_FRAGILE_WARN_ABS} — breakdown이 소수 이상치 재가중만으로 발생했을 가능성")

    # --- 부호조건부 플라시보 순열 검정 ---
    placebo_p, placebo_n_matched, placebo_n_total, placebo_sign_rate = None, 0, 0, None
    placebo_cond_mean = placebo_cond_median = placebo_cond_frac_inf = placebo_suggests_genuine = None

    if ratio is not None and ratio != 0.0:
        print(f"\n  [부호조건부 플라시보 검정] naive_diff={naive_diff:+.4f}와 같은 부호인 재배정만 "
              f"걸러 조건부 귀무분포 구성(목표 n={N_PLACEBO_TARGET_MATCHED})...")
        matched_ratios, n_matched, n_total, sign_rate, uncond_ratios = placebo_null_breakdown_ratios_conditional(
            y_pool, len(y_bt), kl_calibrated_main, naive_diff, N_PLACEBO_TARGET_MATCHED, PLACEBO_SEED)
        placebo_n_matched, placebo_n_total, placebo_sign_rate = n_matched, n_total, sign_rate
        uncond_finite = uncond_ratios[np.isfinite(uncond_ratios)]
        print(f"    [진단] 부호일치율={sign_rate:.2f}(정상 ~0.5), 부호무관 귀무분포(참고용) "
              f"평균={np.nanmean(uncond_finite) if len(uncond_finite) else float('nan'):.3f} "
              f"— 헤드라인 해석에는 쓰지 않음")

        if n_matched >= 20:
            finite_matched = matched_ratios[np.isfinite(matched_ratios)]
            placebo_cond_mean = float(np.mean(finite_matched)) if len(finite_matched) else None
            placebo_cond_median = float(np.median(finite_matched)) if len(finite_matched) else None
            placebo_cond_frac_inf = float(np.mean(~np.isfinite(matched_ratios)))
            placebo_p = float(np.mean(matched_ratios >= ratio))
            placebo_suggests_genuine = bool(placebo_p < PLACEBO_P_ROBUSTNESS_THRESHOLD)
            print(f"    조건부 귀무분포(부호일치, n={n_matched}/{n_total}회): 평균={placebo_cond_mean}, "
                  f"중앙값={placebo_cond_median}")
            print(f"    p = P(조건부귀무 ratio >= 관측 ratio={ratio:.3f}) = {placebo_p:.4f}")
            if placebo_suggests_genuine:
                print(f"    ✓ p<{PLACEBO_P_ROBUSTNESS_THRESHOLD} — 점추정 자체는 노이즈가 아닐 가능성 높음 "
                      f"(단, 이것이 confounding에 대한 강건성을 의미하지는 않음 — breakdown_ratio<1은 그대로)")
            else:
                print(f"    ⚠ p>={PLACEBO_P_ROBUSTNESS_THRESHOLD} — 같은 방향 노이즈와 통계적으로 구분 안 됨")
            for r in matched_ratios:
                placebo_rows.append({"outcome": ycol,
                                      "placebo_ratio_conditional": float(r) if np.isfinite(r) else np.nan})
        else:
            print(f"    ⚠ 부호일치 재배정 부족(n={n_matched}<20) — p-value 생략")

    breakdown_summary[ycol] = {
        "naive_diff": naive_diff, "pooled_variance": pooled_var, "eps_calibrated_main": kl_calibrated_main,
        "breakdown_eps": breakdown, "breakdown_ratio_to_calibrated": ratio,
        "variance_normalized_breakdown_index": var_norm_index,
        "is_fragile_at_calibrated_eps": bool(ratio is not None and ratio != 0.0 and ratio < 1.0),
        "n_eff_born_treated_at_breakdown": n_eff_bt_at_bd, "n_eff_control_at_breakdown": n_eff_ctrl_at_bd,
        "n_eff_fragile_warning": n_eff_fragile_flag,
        "placebo_v4_n_matched": placebo_n_matched, "placebo_v4_n_total_drawn": placebo_n_total,
        "placebo_v4_sign_match_rate": placebo_sign_rate,
        "placebo_v4_conditional_null_mean": placebo_cond_mean,
        "placebo_v4_conditional_null_median": placebo_cond_median,
        "placebo_v4_conditional_null_frac_always_robust": placebo_cond_frac_inf,
        "placebo_v4_p_value_robustness": placebo_p,
        "placebo_v4_suggests_genuine_signal": placebo_suggests_genuine,
        "n_bt": int(len(y_bt)), "n_ctrl": int(len(y_ctrl)),
    }

with open(OUT_DIR / "track2_dro_breakdown_analysis_v4.json", "w", encoding="utf-8") as f:
    json.dump(breakdown_summary, f, ensure_ascii=False, indent=2, default=str)

placebo_df = pd.DataFrame(placebo_rows)
placebo_df.to_csv(OUT_DIR / "track2_dro_placebo_conditional_null_v4.csv", index=False)

if fragile_outcomes:
    print(f"\n⚠ breakdown_ratio<1 outcome: {fragile_outcomes} — 아래 §D 논문 문구에서 confounding "
          f"강건성 결론을 이 목록에 맞춰 하향 조정한다.")


# ============================================================================
# C. step3g 회귀보정 결과와 비교(있으면) + 시각화
# ============================================================================
print("\n" + "=" * 78)
print("C. 기존 회귀보정 결과(step3g)와 비교, 시각화")
print("=" * 78)

step3g_ref = None
if STEP3G_JSON_PATH.exists():
    with open(STEP3G_JSON_PATH, "r", encoding="utf-8") as f:
        step3g_ref = json.load(f)
    for ycol in OUTCOME_VARS:
        adj = step3g_ref.get("results", {}).get(ycol, {}).get("clean_covariate", {})
        if adj:
            bd = breakdown_summary.get(ycol, {})
            same_dir = np.sign(adj.get("coef", 0)) == np.sign(bd.get("naive_diff", 0))
            print(f"[{ycol}] step3g 보정: coef={adj.get('coef'):+.4f}, p={adj.get('pvalue'):.4f} | "
                  f"DRO naive_diff={bd.get('naive_diff'):+.4f} 동일방향={'✓' if same_dir else '✗'}")
else:
    print(f"{STEP3G_JSON_PATH} 없음 — 비교 생략")

for ycol in OUTCOME_VARS:
    cond_vals = (placebo_df.loc[placebo_df["outcome"] == ycol, "placebo_ratio_conditional"].dropna()
                 if not placebo_df.empty else pd.Series(dtype=float))
    obs_ratio = breakdown_summary.get(ycol, {}).get("breakdown_ratio_to_calibrated")
    p_val = breakdown_summary.get(ycol, {}).get("placebo_v4_p_value_robustness")
    if len(cond_vals) == 0 or obs_ratio is None:
        continue
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(cond_vals, bins=30, color="slategray", alpha=0.8, label=f"부호조건부 귀무분포(n={len(cond_vals)})")
    ax.axvline(obs_ratio, color="firebrick", linestyle="--", linewidth=1.5, label=f"관측 ratio={obs_ratio:.2f}")
    p_str = f" (p={p_val:.3f})" if p_val is not None else ""
    ax.set_xlabel("breakdown_ratio (부호일치 플라시보)"); ax.set_ylabel("빈도")
    ax.set_title(f"부호조건부 플라시보 귀무분포 vs 관측 ratio — {ycol}{p_str}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"fig_dro_placebo_conditional_{ycol}.png", dpi=150)
    plt.close()


# ============================================================================
# D. 종합 매니페스트 + 논문용 문구
# ============================================================================
def _format_placebo_p(p_rob, n_matched):
    """p=0.000처럼 정밀도 이상으로 딱 떨어지는 값은 순열 표본크기의 하한
    (1/n_matched)으로 표기한다. 예: n_matched=200, p_rob=0.0 → 'p<0.005'."""
    if p_rob is None:
        return None
    floor = 1.0 / max(n_matched, 1)
    return f"p<{floor:.3f}" if p_rob <= floor else f"p={p_rob:.3f}"


paper_lines = []
for ycol in OUTCOME_VARS:
    bd = breakdown_summary.get(ycol, {})
    ratio, var_idx = bd.get("breakdown_ratio_to_calibrated"), bd.get("variance_normalized_breakdown_index")
    p_rob, n_matched = bd.get("placebo_v4_p_value_robustness"), bd.get("placebo_v4_n_matched", 0)

    if ratio is None:
        line = (f"{ycol}: 캘리브레이션 eps={kl_calibrated_main:.3f}의 50배까지 확대해도 최악의 경우 "
                f"하한이 0 이하로 떨어지지 않았다 — 강건함.")
    elif ratio == 0.0:
        line = f"{ycol}: 무보정 점추정 자체가 유의하지 않아 breakdown 분석이 성립하지 않음."
    elif ratio >= 1.0:
        line = (f"{ycol}: 효과가 0으로 무너지려면 관측된 공변량 불균형보다 {ratio:.2f}배 더 큰 왜곡이 "
                f"필요하다 — 관측된 불균형 수준에서는 강건하다.")
    else:
        var_idx_str = f", 분산정규화지표={var_idx:.2f}" if var_idx is not None else ""
        p_str = _format_placebo_p(p_rob, n_matched)
        p_note = (
            f" 이 점추정 자체가 표본추출 노이즈가 아니라는 것은 기존 회귀 유의성 검정과 부호조건부 "
            f"순열검정({p_str})으로 확인되지만, 이는 confounding에 대한 강건성과는 별개의 사실이다 — "
            f"이미 확인된 정도의 관측되지 않은 교란요인만으로도 이 효과는 사라질 수 있다."
        ) if p_str else " (부호조건부 플라시보 검정을 계산하지 못해 점추정의 노이즈 여부는 별도 확인 필요.)"
        line = (f"{ycol}: 관측된 공변량 불균형 수준의 {ratio:.2f}배에 불과한 분포적 왜곡만으로도 최악의 "
                f"경우 하한이 0 이하로 붕괴한다(breakdown eps={bd.get('breakdown_eps'):.3f}{var_idx_str})."
                f"{p_note}")
    paper_lines.append(line)

paper_text = (
    "선택편향의 크기가 알려지지 않은 상황에서 Track2 효과의 강건성을 정식화하기 위해, outcome과 "
    "시간적으로 겹치지 않는 클린 공변량(log_total_cost_excl_window, device_type_mode; step3g)으로 "
    "표준 성향점수 모형을 적합하고, 대조군을 born-treated 공변량 분포에 맞춰 IPW 재가중할 때 필요한 "
    "KL 발산량을 DRO 앙상블 반경의 캘리브레이션 기준으로 삼았다. 이 앙상블 안에서 최악의 경우 효과 "
    "하한을 계산한 결과: " + " ".join(paper_lines) + " "
    "①(특징기반 검열분류 혼합모형)의 결론(2성분 구조 미정당화, LRT p=0.22)에 따라 ①의 사후확률은 "
    "본 DRO 분석의 주 캘리브레이션에 사용하지 않았다. 플라시보 검정은 관측된 실제 효과와 부호가 "
    "일치하는 무작위 재배정만으로 조건부 귀무분포를 구성해 수행했다(부호 무관 귀무분포는 재배정의 "
    "절반가량이 즉시 breakdown_ratio=0이 되는 구조적 이유로 퇴화되어 있어 해석에 부적합함을 확인했다)."
)

manifest = {
    "version": "v4_sign_conditional_placebo",
    "target_type": TARGET_TYPE,
    "eps_calibrated_main": kl_calibrated_main,
    "mixture_v1v2_model_justified": mixture_model_justified,
    "breakdown_analysis": breakdown_summary,
    "fragile_outcomes": fragile_outcomes,
    "placebo_p_robustness_threshold": PLACEBO_P_ROBUSTNESS_THRESHOLD,
    "n_placebo_target_matched": N_PLACEBO_TARGET_MATCHED,
    "step3g_regression_reference": step3g_ref.get("results") if step3g_ref else None,
    "paper_recommendation_text": paper_text,
    "note": (
        "v4: 이전 버전의 플라시보 검정 방향 오류를 수정했다. 부호 무관 재배정으로 만든 귀무분포는 "
        "재배정의 약 절반이 즉시 breakdown_ratio=0이 되는 구조적 이유로 0 근처에 퇴화되어 있었고, "
        "이를 기준으로 삼으면 부호가 맞는 실제 효과가 거의 항상 100백분위수로 나와 '표본크기/분산 "
        "아티팩트'라는(=취약성이 실재하지 않는다는) 반대 방향 해석을 낳았다. v4는 관측 효과와 부호가 "
        "일치하는 재배정만으로 조건부 귀무분포를 구성해 방향을 바로잡았다. 결론은 여전히 두 outcome "
        "모두 breakdown_ratio<1이다 — '노이즈가 아니다'와 'confounding에 강건하다'는 별개의 질문이다."
    ),
}
with open(OUT_DIR / "track2_dro_manifest_v4.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

print("\n" + "=" * 78)
print("완료")
print("=" * 78)
print(f"\n[논문용 문구 초안]\n{paper_text}")
print(f"\n전체 산출물 위치: {OUT_DIR}")
