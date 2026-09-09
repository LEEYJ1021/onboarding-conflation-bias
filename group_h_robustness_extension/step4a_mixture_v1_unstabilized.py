"""
================================================================================
step4a_mixture_v1_unstabilized.py
Feature-Based Censored Classification v1 — Born-Treated 이분법의 확률적 재정식화
(unstabilized 최초 시도 — 분리(separation) 실패를 투명하게 기록하기 위해 보존)
================================================================================
동기:
    기존 파이프라인(step3c/step2e/step2f)의 gap_days<=threshold 하드 컷오프를
    Ding, Rong & Huh (2024, M&SOM) "Feature-Based Inventory Control with
    Censored Demand"의 아이디어를 차용해 확률적으로 재정식화한다:

      - 잠재계층 Z_i ∈ {0=born-treated, 1=true switcher}
      - 계층배정: P(Z_i=0|X_i) = sigmoid(X_i'β)  (로지스틱 게이팅)
      - 계층조건부 gap_days 분포: Z=0 → Geometric(p0), Z=1 → NegBinom(r1,p1)
      - EM으로 (β,p0,r1,p1) 추정 → 연속 사후확률 P(Z_i=0|gap_i,X_i) 부여

    이 스크립트(v1)는 페널티 없는 최초 시도다. 실행 결과 로지스틱 게이트가
    campaign_type=2(18/19가 born-treated) 같은 희소·불균형 셀에서 준완전분리
    (quasi-complete separation)를 일으켰고(β 95% CI가 [+0.35,+22.58] 등으로
    발산), Geometric 성분이 p0=0.9999로 경계해에 수렴했다. 이 실패 자체가
    step4b(v2, ridge 안정화)의 필요성을 정당화하는 진단적 증거이므로 폐기하지
    않고 그대로 보존한다 — README §23.1, Table 9 "v1" 열의 근거.

한계:
    - 혼합모형 성분개수(1개 vs 2개) LR 검정은 표준 정칙조건을 위반하므로
      (McLachlan & Peel, 2000) 점근 카이제곱 대신 모수적 부트스트랩을 쓴다.
    - N=151 (4개 캠페인유형 풀링)로 표본이 작아 공변량을 최소한으로 유지한다
      (campaign_type, device_type_mode, reg_month).

입력:
    step2_treatment_output/all_types/born_treated_diagnosis_type{T}.csv
    df_analysis_master.csv (device_type_mode)
    step3_stack_level_att_PRIMARY_type6_{outcome}.csv (Track1 재가중용)
    step3d_track2_born_treated_comparison_type6.csv (Track2 재가중용)

출력 (censored_mixture_output/):
    pooled_gap_features.csv, mixture_model_fit.json,
    hard_vs_soft_classification.csv, posterior_probabilities.csv,
    lrt_bootstrap_null_distribution.csv, parameter_bootstrap_ci.json,
    track1_soft_reweighted_type6.json, track2_soft_reweighted_type6.json,
    fig_posterior_vs_gap.png, fig_lrt_null_distribution.png
================================================================================
"""
import os
import re
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
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
ALL_TYPES_DIR = STEP2_DIR / "all_types"
OUT_DIR = Path(os.environ.get("CENSORED_OUT", str(STEP2_DIR / "censored_mixture_output")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_PATH = ROOT / "df_analysis_master.csv"

INIT_THRESHOLD = int(os.environ.get("INIT_HARD_THRESHOLD_DAYS", "1"))  # EM 워밍스타트용, 최종판정 기준 아님
MAX_EM_ITER = int(os.environ.get("MAX_EM_ITER", "300"))
EM_TOL = float(os.environ.get("EM_TOL", "1e-7"))
N_BOOT_LRT = int(os.environ.get("N_BOOT_LRT", "150"))
N_BOOT_PARAM = int(os.environ.get("N_BOOT_PARAM", "150"))
SEED = int(os.environ.get("MIXTURE_SEED", "20260908"))
rng_master = np.random.default_rng(SEED)

TARGET_TYPE_FOR_APPLICATION = int(os.environ.get("TARGET_TYPE_APPLICATION", "6"))

if not ALL_TYPES_DIR.exists():
    raise FileNotFoundError(
        f"{ALL_TYPES_DIR} 없음 — 먼저 step2e_all_types_generalization.py를 실행하세요."
    )
if not MASTER_PATH.exists():
    raise FileNotFoundError(f"{MASTER_PATH} 없음")


# ============================================================================
# A. 데이터 로드 — 전체 유형 풀링 + 구조적 공변량 결합
# ============================================================================
print("=" * 78)
print("A. 데이터 로드 및 특징(feature) 구성")
print("=" * 78)

detail_files = sorted(ALL_TYPES_DIR.glob("born_treated_diagnosis_type*.csv"))
if not detail_files:
    raise FileNotFoundError(f"{ALL_TYPES_DIR}에 born_treated_diagnosis_type*.csv 없음")

frames = []
for f in detail_files:
    m = re.search(r"type(\d+)\.csv$", f.name)
    if not m:
        continue
    T = int(m.group(1))
    df = pd.read_csv(f)
    df["campaign_type"] = T
    frames.append(df)

pooled = pd.concat(frames, ignore_index=True)
pooled["raw_panel_date_min"] = pd.to_datetime(pooled["raw_panel_date_min"])
pooled = pooled.dropna(subset=["gap_days"]).copy()
pooled["gap_days"] = pooled["gap_days"].astype(int)
pooled["reg_month"] = pooled["raw_panel_date_min"].dt.to_period("M").astype(str)

print(f"풀링된 도입고객(전체 유형): {len(pooled)}명")
print(pooled.groupby("campaign_type").size().rename("n").to_string())

master = pd.read_csv(MASTER_PATH)
level_cols = ["customer_id"] + [c for c in ["device_type_mode"] if c in master.columns]
level_data = master.drop_duplicates("customer_id")[level_cols]
pooled = pooled.merge(level_data, on="customer_id", how="left")
if "device_type_mode" in pooled.columns:
    pooled["device_type_mode"] = pooled["device_type_mode"].fillna("unknown")
else:
    pooled["device_type_mode"] = "unknown"

pooled["hard_label_born_treated"] = (pooled["gap_days"] <= INIT_THRESHOLD).astype(int)

feat_path = OUT_DIR / "pooled_gap_features.csv"
pooled.to_csv(feat_path, index=False)
print(f"\n특징 테이블 저장: {feat_path}")


def build_design_matrix(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    X = pd.get_dummies(df[categorical_cols].astype(str), drop_first=True)
    X = sm.add_constant(X, has_constant="add")
    return X.astype(float)


CATEGORICAL_COVARIATES = ["campaign_type", "device_type_mode", "reg_month"]
X_design = build_design_matrix(pooled, CATEGORICAL_COVARIATES)
gap = pooled["gap_days"].to_numpy()
print(f"\n설계행렬 형태: {X_design.shape} (열: {list(X_design.columns)})")


# ============================================================================
# B. EM 알고리즘 — 특징기반 2성분 혼합모형 (페널티 없음)
# ============================================================================
print("\n" + "=" * 78)
print("B. 특징기반 2성분 혼합모형(EM) 적합 — unstabilized")
print("=" * 78)


def _safe_log_sigmoid(logit):
    return -np.logaddexp(0.0, -logit)


def e_step(gap_arr, X_arr, beta, p0, r1, p1):
    logit = X_arr @ beta
    log_pi0 = _safe_log_sigmoid(logit)
    log_pi1 = _safe_log_sigmoid(-logit)
    log_f0 = stats.geom.logpmf(gap_arr + 1, p0)
    log_f1 = stats.nbinom.logpmf(gap_arr, r1, p1)
    log_joint0 = log_pi0 + log_f0
    log_joint1 = log_pi1 + log_f1
    log_norm = np.logaddexp(log_joint0, log_joint1)
    resp0 = np.clip(np.exp(log_joint0 - log_norm), 1e-10, 1 - 1e-10)
    return resp0, float(np.sum(log_norm))


def m_step_logistic(X_arr, resp0, beta_init):
    try:
        glm_res = sm.GLM(endog=resp0, exog=X_arr, family=sm.families.Binomial()).fit(
            start_params=beta_init, maxiter=100
        )
        return np.asarray(glm_res.params)
    except Exception:
        return beta_init


def m_step_geometric(gap_arr, resp0):
    num = np.sum(resp0)
    den = np.sum(resp0 * (gap_arr + 1))
    p0 = num / den if den > 0 else 0.5
    return float(np.clip(p0, 1e-4, 1 - 1e-4))


def m_step_negbinom(gap_arr, resp1, r_init, p_init):
    def neg_ll(params):
        r, p = params
        if r <= 1e-6 or not (1e-6 < p < 1 - 1e-6):
            return np.inf
        return -np.sum(resp1 * stats.nbinom.logpmf(gap_arr, r, p))

    res = minimize(neg_ll, x0=[max(r_init, 0.5), np.clip(p_init, 0.05, 0.95)],
                    method="L-BFGS-B", bounds=[(1e-3, None), (1e-3, 1 - 1e-3)])
    return float(res.x[0]), float(res.x[1])


def fit_em(gap_arr, X_arr, init_threshold=INIT_THRESHOLD, max_iter=MAX_EM_ITER,
           tol=EM_TOL, verbose=False):
    hard0 = (gap_arr <= init_threshold).astype(float)
    hard0_smooth = hard0 * 0.9 + 0.05
    try:
        beta = sm.GLM(endog=hard0_smooth, exog=X_arr, family=sm.families.Binomial()).fit().params.to_numpy()
    except Exception:
        beta = np.zeros(X_arr.shape[1])

    mask0 = hard0 > 0.5
    mask1 = ~mask0
    p0 = m_step_geometric(gap_arr[mask0], np.ones(mask0.sum())) if mask0.sum() > 0 else 0.9
    if mask1.sum() > 1:
        m_, v_ = gap_arr[mask1].mean(), gap_arr[mask1].var()
        p1_init, r1_init = (m_ / v_, m_ * (m_ / v_) / (1 - m_ / v_)) if v_ > m_ > 0 else (0.3, 2.0)
    else:
        p1_init, r1_init = 0.3, 2.0
    r1, p1 = m_step_negbinom(gap_arr[mask1] if mask1.sum() > 0 else gap_arr,
                              np.ones(max(mask1.sum(), 1)), r1_init, p1_init)

    prev_ll, it = -np.inf, 0
    for it in range(max_iter):
        resp0, loglik = e_step(gap_arr, X_arr, beta, p0, r1, p1)
        resp1 = 1 - resp0
        beta = m_step_logistic(X_arr, resp0, beta)
        p0 = m_step_geometric(gap_arr, resp0)
        r1, p1 = m_step_negbinom(gap_arr, resp1, r1, p1)
        if verbose and it % 20 == 0:
            print(f"  iter={it:>3} loglik={loglik:.4f}")
        if abs(loglik - prev_ll) < tol:
            break
        prev_ll = loglik

    resp0_final, loglik_final = e_step(gap_arr, X_arr, beta, p0, r1, p1)
    n_params = len(beta) + 1 + 2
    aic = 2 * n_params - 2 * loglik_final
    bic = n_params * np.log(len(gap_arr)) - 2 * loglik_final
    return {"beta": beta, "p0": p0, "r1": r1, "p1": p1, "loglik": loglik_final,
            "n_iter": it + 1, "resp0": resp0_final, "n_params": n_params, "aic": aic, "bic": bic}


fit_result = fit_em(gap, X_design.to_numpy(), verbose=True)
print(f"\nEM 수렴: {fit_result['n_iter']}회, 로그가능도={fit_result['loglik']:.4f}")
print(f"p0={fit_result['p0']:.4f}  r1,p1=({fit_result['r1']:.4f}, {fit_result['p1']:.4f})")
print(f"AIC={fit_result['aic']:.2f}, BIC={fit_result['bic']:.2f}")

beta_names = list(X_design.columns)
print("\n로지스틱 게이트 계수(β):")
for name, b in zip(beta_names, fit_result["beta"]):
    print(f"  {name:<30} {b:+.4f}")


# ============================================================================
# C. 하드 threshold vs 소프트 사후확률 비교
# ============================================================================
print("\n" + "=" * 78)
print("C. 하드 threshold(gap<=1) vs 소프트 사후확률 비교")
print("=" * 78)

pooled["posterior_born_treated_prob"] = fit_result["resp0"]
pooled["posterior_true_switcher_prob"] = 1 - fit_result["resp0"]

AMBIGUOUS_LO, AMBIGUOUS_HI = 0.2, 0.8
pooled["is_ambiguous"] = pooled["posterior_born_treated_prob"].between(AMBIGUOUS_LO, AMBIGUOUS_HI)
pooled["hard_soft_disagree"] = (
    (pooled["hard_label_born_treated"] == 1) & (pooled["posterior_born_treated_prob"] < 0.5)
) | (
    (pooled["hard_label_born_treated"] == 0) & (pooled["posterior_born_treated_prob"] >= 0.5)
)

n_ambig = int(pooled["is_ambiguous"].sum())
n_disagree = int(pooled["hard_soft_disagree"].sum())
print(f"애매 구간({AMBIGUOUS_LO}~{AMBIGUOUS_HI}) 고객: {n_ambig}명 | 하드/소프트 불일치: {n_disagree}명")
if n_ambig == 0:
    print("⚠ n_ambiguous=0 — Geometric 성분이 gap=0 근처 point mass로 붕괴했을 가능성 (아래 p0 확인).")

compare_path = OUT_DIR / "hard_vs_soft_classification.csv"
pooled[["customer_id", "campaign_type", "gap_days", "n_pre_treated_days", "is_valid_stack",
        "hard_label_born_treated", "posterior_born_treated_prob", "is_ambiguous",
        "hard_soft_disagree"]].to_csv(compare_path, index=False)
pooled.to_csv(OUT_DIR / "posterior_probabilities.csv", index=False)
print(f"저장: {compare_path}")

fig, ax = plt.subplots(figsize=(8, 5))
colors = pooled["campaign_type"].map(
    {t: c for t, c in zip(sorted(pooled["campaign_type"].unique()), plt.cm.tab10.colors)}
)
ax.scatter(pooled["gap_days"], pooled["posterior_born_treated_prob"], c=colors, alpha=0.75,
           edgecolor="white", s=60)
ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
ax.axvline(INIT_THRESHOLD + 0.5, color="firebrick", linestyle="--", linewidth=1,
           label=f"기존 하드 threshold(gap<={INIT_THRESHOLD})")
ax.set_xlabel("gap_days")
ax.set_ylabel("사후확률 P(born-treated | gap, X)")
ax.set_title("v1(unstabilized) 특징기반 혼합모형")
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_posterior_vs_gap.png", dpi=150)
plt.close()


# ============================================================================
# D. 단일성분(null) vs 2성분 혼합모형 — 부트스트랩 LR 검정
# ============================================================================
print("\n" + "=" * 78)
print("D. 모형검증: 단일성분(null) vs 2성분 — 부트스트랩 LR 검정")
print("=" * 78)
print("주의: 성분개수 검정은 표준 정칙조건을 위반하므로(McLachlan & Peel, 2000) "
      "점근근사 대신 모수적 부트스트랩으로 귀무분포를 직접 시뮬레이션한다.\n")


def fit_null_nb(gap_arr):
    m_, v_ = gap_arr.mean(), gap_arr.var()
    p_init, r_init = (m_ / v_, m_ * (m_ / v_) / (1 - m_ / v_)) if v_ > m_ > 0 else (0.3, 2.0)

    def neg_ll(params):
        r, p = params
        if r <= 1e-6 or not (1e-6 < p < 1 - 1e-6):
            return np.inf
        return -np.sum(stats.nbinom.logpmf(gap_arr, r, p))

    res = minimize(neg_ll, x0=[r_init, p_init], method="L-BFGS-B",
                    bounds=[(1e-3, None), (1e-3, 1 - 1e-3)])
    return {"r": float(res.x[0]), "p": float(res.x[1]), "loglik": float(-res.fun)}


null_fit = fit_null_nb(gap)
lr_observed = 2 * (fit_result["loglik"] - null_fit["loglik"])
print(f"귀무모형 로그가능도={null_fit['loglik']:.4f} | 관측 LR={lr_observed:.4f}")


def parametric_bootstrap_lrt(X_arr, null_params, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = X_arr.shape[0]
    lr_boot = []
    for _ in range(n_boot):
        gap_sim = stats.nbinom.rvs(null_params["r"], null_params["p"], size=n, random_state=rng)
        try:
            null_b = fit_null_nb(gap_sim)
            mix_b = fit_em(gap_sim, X_arr, max_iter=150, tol=1e-6)
            lr_boot.append(max(2 * (mix_b["loglik"] - null_b["loglik"]), 0.0))
        except Exception:
            continue
    return np.array(lr_boot)


lr_null_dist = parametric_bootstrap_lrt(X_design.to_numpy(), null_fit, N_BOOT_LRT,
                                         rng_master.integers(0, 2**32 - 1))
n_success = len(lr_null_dist)
lrt_pvalue = float(np.mean(lr_null_dist >= lr_observed)) if n_success > 0 else np.nan
print(f"성공 반복: {n_success}/{N_BOOT_LRT} | 부트스트랩 LRT p-value = {lrt_pvalue:.4f}")

pd.DataFrame({"lr_boot": lr_null_dist}).to_csv(OUT_DIR / "lrt_bootstrap_null_distribution.csv", index=False)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
if n_success > 0:
    ax.hist(lr_null_dist, bins=30, color="steelblue", alpha=0.8, label="부트스트랩 귀무분포(LR)")
ax.axvline(lr_observed, color="firebrick", linestyle="--", linewidth=1.5, label=f"관측 LR={lr_observed:.2f}")
ax.set_xlabel("LR 통계량")
ax.set_ylabel("빈도")
ax.set_title(f"1성분 vs 2성분 — 부트스트랩 LR 검정 (p={lrt_pvalue:.4f})")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_lrt_null_distribution.png", dpi=150)
plt.close()


# ============================================================================
# E. 파라미터 불확실성 — 비모수 부트스트랩 (분리 진단 목적)
# ============================================================================
print("\n" + "=" * 78)
print(f"E. 파라미터 불확실성 — 비모수 부트스트랩(B={N_BOOT_PARAM})")
print("=" * 78)

n_obs = len(gap)
boot_betas, boot_p0, boot_r1, boot_p1 = [], [], [], []
rng_param = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
for _ in range(N_BOOT_PARAM):
    idx = rng_param.integers(0, n_obs, size=n_obs)
    try:
        fit_b = fit_em(gap[idx], X_design.to_numpy()[idx], max_iter=150, tol=1e-6)
        boot_betas.append(fit_b["beta"]); boot_p0.append(fit_b["p0"])
        boot_r1.append(fit_b["r1"]); boot_p1.append(fit_b["p1"])
    except Exception:
        continue

n_boot_success = len(boot_p0)
print(f"성공 반복: {n_boot_success}/{N_BOOT_PARAM}")

param_ci = {}
if n_boot_success >= 30:
    boot_betas_arr = np.array(boot_betas)
    beta_ci = {}
    for j, name in enumerate(beta_names):
        lo, hi = np.percentile(boot_betas_arr[:, j], [2.5, 97.5])
        beta_ci[name] = {"point": float(fit_result["beta"][j]), "ci_lo": float(lo), "ci_hi": float(hi)}
        flag = "  ⚠ 발산 의심(분리)" if (hi - lo) > 5.0 else ""
        print(f"  β[{name:<28}] 점추정={fit_result['beta'][j]:+.4f}, 95% CI=[{lo:+.4f}, {hi:+.4f}]{flag}")
    p0_lo, p0_hi = np.percentile(boot_p0, [2.5, 97.5])
    print(f"  p0 점추정={fit_result['p0']:.4f}, 95% CI=[{p0_lo:.4f}, {p0_hi:.4f}]")
    param_ci = {"beta": beta_ci, "p0": {"point": fit_result["p0"], "ci_lo": float(p0_lo), "ci_hi": float(p0_hi)}}

with open(OUT_DIR / "parameter_bootstrap_ci.json", "w", encoding="utf-8") as f:
    json.dump(param_ci, f, ensure_ascii=False, indent=2, default=str)


# ============================================================================
# F. [참고, 미검증 모형 위에서의 시연] Type6 소프트 재가중
#    주의: 아래는 D단계 검증 없이 계산된 것이므로 본문 근거로 사용하지 않는다.
#    (step4b의 MODEL_JUSTIFIED 게이팅으로 정식 대체된다.)
# ============================================================================
print("\n" + "=" * 78)
print(f"F. [시연 전용, 미검증] Type{TARGET_TYPE_FOR_APPLICATION} 소프트 재가중")
print("=" * 78)

type_mask = pooled["campaign_type"] == TARGET_TYPE_FOR_APPLICATION
type_posterior = pooled.loc[type_mask, ["customer_id", "gap_days", "posterior_born_treated_prob",
                                         "posterior_true_switcher_prob", "hard_label_born_treated",
                                         "n_pre_treated_days"]].copy()

OUTCOME_FILES = {
    "log_spend_safe": STEP2_DIR / f"step3_stack_level_att_PRIMARY_type{TARGET_TYPE_FOR_APPLICATION}_log_spend_safe.csv",
    "n_campaign_types_active": STEP2_DIR / f"step3_stack_level_att_PRIMARY_type{TARGET_TYPE_FOR_APPLICATION}_n_campaign_types_active.csv",
}
track1_results = {}
for outcome, path in OUTCOME_FILES.items():
    if not path.exists():
        continue
    merged = pd.read_csv(path).merge(type_posterior, on="customer_id", how="inner")
    if merged.empty:
        continue
    hard_mask = merged["hard_label_born_treated"] == 0
    hard_mean = float(merged.loc[hard_mask, "att_i"].mean()) if hard_mask.sum() else np.nan
    w, x = merged["posterior_true_switcher_prob"].to_numpy(), merged["att_i"].to_numpy()
    soft_mean = float(np.sum(w * x) / np.sum(w)) if np.sum(w) > 0 else np.nan
    n_eff = float((np.sum(w) ** 2) / np.sum(w ** 2)) if np.sum(w ** 2) > 0 else np.nan
    print(f"  [{outcome}] 하드 n={int(hard_mask.sum())} mean={hard_mean:+.4f} | "
          f"소프트 n_eff={n_eff:.2f} weighted_mean={soft_mean:+.4f}")
    track1_results[outcome] = {"hard_mean_att": hard_mean, "soft_weighted_mean_att": soft_mean,
                                "soft_n_eff_kish": n_eff}

with open(OUT_DIR / f"track1_soft_reweighted_type{TARGET_TYPE_FOR_APPLICATION}.json", "w", encoding="utf-8") as f:
    json.dump(track1_results, f, ensure_ascii=False, indent=2, default=str)

track2_path = STEP2_DIR / f"step3d_track2_born_treated_comparison_type{TARGET_TYPE_FOR_APPLICATION}.csv"
track2_results = {}
if track2_path.exists():
    t2 = pd.read_csv(track2_path).merge(
        type_posterior[["customer_id", "posterior_born_treated_prob"]], on="customer_id", how="left")
    for ycol in ["log_spend_safe", "n_campaign_types_active"]:
        bt_col, ctrl_col = f"{ycol}_born_treated", f"{ycol}_control_mean"
        if bt_col not in t2.columns:
            continue
        sub = t2.dropna(subset=[bt_col, ctrl_col, "posterior_born_treated_prob"])
        if sub.empty:
            continue
        w = sub["posterior_born_treated_prob"].to_numpy()
        diff = sub[bt_col].to_numpy() - sub[ctrl_col].to_numpy()
        soft_diff = float(np.sum(w * diff) / np.sum(w)) if np.sum(w) > 0 else np.nan
        print(f"  [{ycol}] 소프트 재가중 격차={soft_diff:+.3f} (n={len(sub)})")
        track2_results[ycol] = {"soft_weighted_diff": soft_diff, "n_hard": int(len(sub))}

with open(OUT_DIR / f"track2_soft_reweighted_type{TARGET_TYPE_FOR_APPLICATION}.json", "w", encoding="utf-8") as f:
    json.dump(track2_results, f, ensure_ascii=False, indent=2, default=str)


# ============================================================================
# G. 종합 매니페스트
# ============================================================================
manifest = {
    "version": "v1_unstabilized",
    "n_pooled_adopters": int(len(pooled)),
    "covariates_used": CATEGORICAL_COVARIATES,
    "em_fit": {"beta_names": beta_names, "beta": fit_result["beta"].tolist(),
               "p0": fit_result["p0"], "r1": fit_result["r1"], "p1": fit_result["p1"],
               "loglik": fit_result["loglik"], "aic": fit_result["aic"], "bic": fit_result["bic"]},
    "lrt_bootstrap": {"lr_observed": lr_observed, "p_value": lrt_pvalue if pd.notna(lrt_pvalue) else None},
    "hard_vs_soft": {"n_ambiguous": n_ambig, "n_disagree": n_disagree},
    "known_issues": [
        "로지스틱 게이트에 페널티 없음 — 희소·불균형 셀(campaign_type=2 등)에서 준완전분리 발생",
        "p0가 경계해(≈1)에 근접 — Geometric 성분이 gap=0 point mass로 붕괴",
        "이 파일의 F단계 결과는 시연용일 뿐 본문 근거로 사용하지 않음 — step4b 참조",
    ],
    "note": "이 스크립트는 안정화 이전 버전을 투명하게 보존하기 위한 것이다. "
            "정식 분석은 step4b_mixture_v2_stabilized.py를 따른다.",
}
with open(OUT_DIR / "mixture_model_fit.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

print("\n" + "=" * 78)
print("완료 (v1, unstabilized) — 다음 단계: step4b_mixture_v2_stabilized.py")
print("=" * 78)
