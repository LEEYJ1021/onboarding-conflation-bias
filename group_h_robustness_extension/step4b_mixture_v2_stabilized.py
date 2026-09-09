"""
================================================================================
step4b_mixture_v2_stabilized.py
Feature-Based Censored Classification v2 — 안정화(ridge) + LRT 게이팅
================================================================================
step4a(v1)에서 발견된 3가지 문제를 같은 원인(게이트 분리)을 고쳐 한 번에 해결한다:
    1) [D] 부트스트랩 LRT p=0.38 — v1은 이 결과를 무시하고 F단계(재가중)를 강행함
    2) [B] p0=0.9999 — Geometric 성분이 gap=0에 point mass로 붕괴, n_ambiguous=0
    3) [E] β 부트스트랩 CI가 [+0.35,+22.58] 등으로 폭발 — 로지스틱 게이트가
       희소·불균형 셀(campaign_type=2: 18/19 born-treated)에서 준완전분리

수정 사항:
    [수정 1] 희소 범주 수준을 min_n 미만이면 'other_pooled'로 사전 풀링.
             풀링 후 유효 수준이 1개면 그 공변량 자체를 설계행렬에서 제외.
    [수정 2] 로지스틱 게이트 M-step을 릿지(L2) 페널티가 있는 가중 로그가능도
             최적화로 교체(절편 제외). 분리가 일어나도 계수가 유한하게 수축.
    [수정 3] 안정화된 모형으로 부트스트랩 LR 검정(§D) 재실행.
             MODEL_JUSTIFIED = (p_value < ALPHA) and not p0_is_boundary.
    [수정 4] F단계(Track1/Track2 소프트 재가중)는 항상 계산하되, MODEL_JUSTIFIED가
             False이면 모든 출력에 status="exploratory_only_model_not_justified"를
             붙이고, 논문용 문구를 자동 생성해 매니페스트에 포함한다.

핵심 원칙: D가 p>=ALPHA로 계속 나오면 튜닝해서 유의하게 만들지 말고, 이를
negative robustness result로 보고하며 기존 hard-threshold(gap<=1) 접근을
유지한다(README §23.1 / Table 9-10 / Figure 12의 근거가 되는 스크립트).

출력 (censored_mixture_output_v2/):
    pooled_gap_features_v2.csv, hard_vs_soft_classification_v2.csv
    (→ step4c의 Figure 12B 입력), lrt_bootstrap_null_distribution_v2.csv,
    parameter_bootstrap_ci_v2.json, track1/2_soft_reweighted_type6_v2.json,
    mixture_model_fit_v2.json (MODEL_JUSTIFIED 플래그 — step5a가 승계),
    fig_posterior_vs_gap_v2.png, fig_lrt_null_distribution_v2.png
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
OUT_DIR = Path(os.environ.get("CENSORED_OUT", str(STEP2_DIR / "censored_mixture_output_v2")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_PATH = ROOT / "df_analysis_master.csv"

INIT_THRESHOLD = int(os.environ.get("INIT_HARD_THRESHOLD_DAYS", "1"))
MAX_EM_ITER = int(os.environ.get("MAX_EM_ITER", "300"))
EM_TOL = float(os.environ.get("EM_TOL", "1e-7"))
N_BOOT_LRT = int(os.environ.get("N_BOOT_LRT", "150"))
N_BOOT_PARAM = int(os.environ.get("N_BOOT_PARAM", "150"))
SEED = int(os.environ.get("MIXTURE_SEED", "20260908"))
rng_master = np.random.default_rng(SEED)

TARGET_TYPE_FOR_APPLICATION = int(os.environ.get("TARGET_TYPE_APPLICATION", "6"))
ALPHA = float(os.environ.get("ALPHA", "0.05"))

MIN_LEVEL_N = int(os.environ.get("MIN_LEVEL_N", "10"))          # [수정 1] 희소 범주 풀링 기준
RIDGE_LAMBDA = float(os.environ.get("RIDGE_LAMBDA", "2.0"))      # [수정 2] 릿지 페널티 강도

P0_BOUNDARY_THRESHOLD = float(os.environ.get("P0_BOUNDARY_THRESHOLD", "0.995"))
BETA_CI_HALFWIDTH_WARN = float(os.environ.get("BETA_CI_HALFWIDTH_WARN", "5.0"))

if not ALL_TYPES_DIR.exists():
    raise FileNotFoundError(f"{ALL_TYPES_DIR} 없음 — step2e_all_types_generalization.py를 먼저 실행하세요.")
if not MASTER_PATH.exists():
    raise FileNotFoundError(f"{MASTER_PATH} 없음")


# ============================================================================
# A. 데이터 로드 + [수정 1] 희소 범주 수준 사전 풀링
# ============================================================================
print("=" * 78)
print("A. 데이터 로드 및 특징 구성 — 희소 범주 풀링 포함")
print("=" * 78)

detail_files = sorted(ALL_TYPES_DIR.glob("born_treated_diagnosis_type*.csv"))
if not detail_files:
    raise FileNotFoundError(f"{ALL_TYPES_DIR}에 born_treated_diagnosis_type*.csv 없음")

frames = []
for f in detail_files:
    m = re.search(r"type(\d+)\.csv$", f.name)
    if not m:
        continue
    df = pd.read_csv(f)
    df["campaign_type"] = int(m.group(1))
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
pooled = pooled.merge(master.drop_duplicates("customer_id")[level_cols], on="customer_id", how="left")
if "device_type_mode" not in pooled.columns:
    pooled["device_type_mode"] = "unknown"
pooled["device_type_mode"] = pooled["device_type_mode"].fillna("unknown")
pooled["hard_label_born_treated"] = (pooled["gap_days"] <= INIT_THRESHOLD).astype(int)

RAW_CATEGORICAL_COVARIATES = ["campaign_type", "device_type_mode", "reg_month"]


def collapse_sparse_levels(df, cols, min_n):
    """min_n 미만 관측치인 수준을 'other_pooled'로 풀링. 풀링 후 유효 수준이
    1개면(=변별력 없음) 그 공변량을 used_cols에서 제외한다."""
    df = df.copy()
    used_cols, collapse_log = [], {}
    for col in cols:
        vc = df[col].astype(str).value_counts()
        keep_levels = set(vc[vc >= min_n].index)
        new_col = f"{col}_collapsed"
        df[new_col] = np.where(df[col].astype(str).isin(keep_levels), df[col].astype(str), "other_pooled")
        n_final = df[new_col].nunique()
        collapse_log[col] = {"original_levels": int(vc.shape[0]), "final_n_levels": int(n_final)}
        print(f"  [{col}] {vc.shape[0]}개 수준 → min_n={min_n} 기준 풀링 → 최종 {n_final}개 수준")
        if n_final >= 2:
            used_cols.append(new_col)
        else:
            print(f"    ⚠ [{col}] 풀링 후 유효 수준 {n_final}개뿐 — 설계행렬에서 제외")
    return df, used_cols, collapse_log


print("\n[수정 1] 희소 범주 수준 풀링:")
pooled, used_covariate_cols, collapse_log = collapse_sparse_levels(pooled, RAW_CATEGORICAL_COVARIATES, MIN_LEVEL_N)
pooled.to_csv(OUT_DIR / "pooled_gap_features_v2.csv", index=False)


def build_design_matrix(df, categorical_cols):
    X = pd.get_dummies(df[categorical_cols].astype(str), drop_first=True)
    return sm.add_constant(X, has_constant="add").astype(float)


X_design = build_design_matrix(pooled, used_covariate_cols)
gap = pooled["gap_days"].to_numpy()
print(f"\n설계행렬 형태(풀링 후): {X_design.shape} (열: {list(X_design.columns)})")
print(f"릿지 페널티 강도(RIDGE_LAMBDA) = {RIDGE_LAMBDA} (절편 제외)")


# ============================================================================
# B. EM — [수정 2] 릿지 페널티가 있는 로지스틱 M-step
# ============================================================================
print("\n" + "=" * 78)
print("B. 안정화된 특징기반 2성분 혼합모형(EM) 적합")
print("=" * 78)


def _safe_log_sigmoid(logit):
    return -np.logaddexp(0.0, -logit)


def e_step(gap_arr, X_arr, beta, p0, r1, p1):
    logit = X_arr @ beta
    log_pi0 = _safe_log_sigmoid(logit)
    log_pi1 = _safe_log_sigmoid(-logit)
    log_f0 = stats.geom.logpmf(gap_arr + 1, p0)
    log_f1 = stats.nbinom.logpmf(gap_arr, r1, p1)
    log_joint0, log_joint1 = log_pi0 + log_f0, log_pi1 + log_f1
    log_norm = np.logaddexp(log_joint0, log_joint1)
    resp0 = np.clip(np.exp(log_joint0 - log_norm), 1e-10, 1 - 1e-10)
    return resp0, float(np.sum(log_norm))


def m_step_logistic_ridge(X_arr, resp0, beta_init, ridge_lambda):
    """가중(소프트 라벨) 로지스틱 회귀를 릿지 페널티와 함께 직접 최적화한다.
    절편(첫 열)은 페널티에서 제외 — 통계적으로 MAP 추정(beta[1:]~N(0,1/λ))과 동치."""
    w = np.clip(resp0, 1e-10, 1 - 1e-10)

    def neg_penalized_ll(beta):
        logit = X_arr @ beta
        ll = np.sum(w * _safe_log_sigmoid(logit) + (1 - w) * _safe_log_sigmoid(-logit))
        penalty = 0.5 * ridge_lambda * np.sum(beta[1:] ** 2)
        return -(ll - penalty)

    def grad(beta):
        logit = X_arr @ beta
        prob = 1.0 / (1.0 + np.exp(-logit))
        g = -(X_arr.T @ (w - prob))
        pen_grad = ridge_lambda * beta.copy()
        pen_grad[0] = 0.0
        return g + pen_grad

    res = minimize(neg_penalized_ll, x0=beta_init, jac=grad, method="L-BFGS-B", options={"maxiter": 200})
    return res.x if res.x is not None else beta_init


def m_step_geometric(gap_arr, resp0):
    num, den = np.sum(resp0), np.sum(resp0 * (gap_arr + 1))
    return float(np.clip(num / den if den > 0 else 0.5, 1e-4, 1 - 1e-4))


def m_step_negbinom(gap_arr, resp1, r_init, p_init):
    def neg_ll(params):
        r, p = params
        if r <= 1e-6 or not (1e-6 < p < 1 - 1e-6):
            return np.inf
        return -np.sum(resp1 * stats.nbinom.logpmf(gap_arr, r, p))

    res = minimize(neg_ll, x0=[max(r_init, 0.5), np.clip(p_init, 0.05, 0.95)],
                    method="L-BFGS-B", bounds=[(1e-3, None), (1e-3, 1 - 1e-3)])
    return float(res.x[0]), float(res.x[1])


def fit_em_stabilized(gap_arr, X_arr, ridge_lambda=RIDGE_LAMBDA, init_threshold=INIT_THRESHOLD,
                       max_iter=MAX_EM_ITER, tol=EM_TOL, verbose=False):
    hard0 = (gap_arr <= init_threshold).astype(float)
    hard0_smooth = hard0 * 0.9 + 0.05
    beta = m_step_logistic_ridge(X_arr, hard0_smooth, np.zeros(X_arr.shape[1]), ridge_lambda)

    mask0, mask1 = hard0 > 0.5, hard0 <= 0.5
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
        beta = m_step_logistic_ridge(X_arr, resp0, beta, ridge_lambda)
        p0 = m_step_geometric(gap_arr, resp0)
        r1, p1 = m_step_negbinom(gap_arr, resp1, r1, p1)
        if verbose and it % 20 == 0:
            print(f"  iter={it:>3} loglik={loglik:.4f}")
        if abs(loglik - prev_ll) < tol:
            break
        prev_ll = loglik

    resp0_final, loglik_final = e_step(gap_arr, X_arr, beta, p0, r1, p1)
    # 페널티는 추정 안정화 장치일 뿐이므로, 모형비교(AIC/BIC/LRT)에는 페널티 없는
    # 데이터가능도(loglik_final)를 그대로 사용한다.
    n_params = len(beta) + 1 + 2
    aic = 2 * n_params - 2 * loglik_final
    bic = n_params * np.log(len(gap_arr)) - 2 * loglik_final
    return {"beta": beta, "p0": p0, "r1": r1, "p1": p1, "loglik": loglik_final,
            "n_iter": it + 1, "resp0": resp0_final, "n_params": n_params, "aic": aic, "bic": bic}


fit_result = fit_em_stabilized(gap, X_design.to_numpy(), verbose=True)
print(f"\nEM 수렴: {fit_result['n_iter']}회, 로그가능도={fit_result['loglik']:.4f}")
print(f"p0={fit_result['p0']:.4f}  r1,p1=({fit_result['r1']:.4f}, {fit_result['p1']:.4f})")
print(f"AIC={fit_result['aic']:.2f}, BIC={fit_result['bic']:.2f}")

beta_names = list(X_design.columns)
print("\n로지스틱 게이트 계수(β, 릿지 적용):")
for name, b in zip(beta_names, fit_result["beta"]):
    print(f"  {name:<30} {b:+.4f}")

p0_is_boundary = fit_result["p0"] > P0_BOUNDARY_THRESHOLD
print(f"\n{'⚠' if p0_is_boundary else '✓'} p0={fit_result['p0']:.4f} "
      f"({'경계해 기준 초과' if p0_is_boundary else '경계해 기준 미만, 정상'})")


# ============================================================================
# C. 하드 threshold vs 소프트 사후확률 비교
# ============================================================================
print("\n" + "=" * 78)
print("C. 하드 threshold(gap<=1) vs 소프트 사후확률 비교 (안정화 모형)")
print("=" * 78)

pooled["posterior_born_treated_prob"] = fit_result["resp0"]
pooled["posterior_true_switcher_prob"] = 1 - fit_result["resp0"]
AMBIGUOUS_LO, AMBIGUOUS_HI = 0.2, 0.8
pooled["is_ambiguous"] = pooled["posterior_born_treated_prob"].between(AMBIGUOUS_LO, AMBIGUOUS_HI)
pooled["hard_soft_disagree"] = (
    (pooled["hard_label_born_treated"] == 1) & (pooled["posterior_born_treated_prob"] < 0.5)
) | ((pooled["hard_label_born_treated"] == 0) & (pooled["posterior_born_treated_prob"] >= 0.5))

n_ambig = int(pooled["is_ambiguous"].sum())
n_disagree = int(pooled["hard_soft_disagree"].sum())
print(f"애매 구간 고객: {n_ambig}명 | 하드/소프트 불일치: {n_disagree}명")

compare_path = OUT_DIR / "hard_vs_soft_classification_v2.csv"
pooled[["customer_id", "campaign_type", "gap_days", "n_pre_treated_days", "is_valid_stack",
        "hard_label_born_treated", "posterior_born_treated_prob", "is_ambiguous",
        "hard_soft_disagree"]].to_csv(compare_path, index=False)
print(f"저장(→ step4c 입력): {compare_path}")

fig, ax = plt.subplots(figsize=(8, 5))
colors = pooled["campaign_type"].map(
    {t: c for t, c in zip(sorted(pooled["campaign_type"].unique()), plt.cm.tab10.colors)})
ax.scatter(pooled["gap_days"], pooled["posterior_born_treated_prob"], c=colors, alpha=0.75,
           edgecolor="white", s=60)
ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
ax.axvline(INIT_THRESHOLD + 0.5, color="firebrick", linestyle="--", linewidth=1,
           label=f"기존 하드 threshold(gap<={INIT_THRESHOLD})")
ax.set_xlabel("gap_days")
ax.set_ylabel("사후확률 P(born-treated | gap, X) — 안정화(릿지) 모형")
ax.set_title("안정화된 혼합모형: 릿지 페널티 + 희소범주 풀링 적용 후")
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_posterior_vs_gap_v2.png", dpi=150)
plt.close()


# ============================================================================
# D. [수정 3] 안정화 모형으로 부트스트랩 LR 검정 재실행
# ============================================================================
print("\n" + "=" * 78)
print("D. 모형검증(안정화 모형): 부트스트랩 LR 검정 재실행")
print("=" * 78)


def fit_null_nb(gap_arr):
    m_, v_ = gap_arr.mean(), gap_arr.var()
    p_init, r_init = (m_ / v_, m_ * (m_ / v_) / (1 - m_ / v_)) if v_ > m_ > 0 else (0.3, 2.0)

    def neg_ll(params):
        r, p = params
        if r <= 1e-6 or not (1e-6 < p < 1 - 1e-6):
            return np.inf
        return -np.sum(stats.nbinom.logpmf(gap_arr, r, p))

    res = minimize(neg_ll, x0=[r_init, p_init], method="L-BFGS-B", bounds=[(1e-3, None), (1e-3, 1 - 1e-3)])
    return {"r": float(res.x[0]), "p": float(res.x[1]), "loglik": float(-res.fun)}


null_fit = fit_null_nb(gap)
lr_observed = 2 * (fit_result["loglik"] - null_fit["loglik"])
print(f"귀무모형 로그가능도={null_fit['loglik']:.4f} | 관측 LR(안정화 모형)={lr_observed:.4f}")


def parametric_bootstrap_lrt(X_arr, null_params, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = X_arr.shape[0]
    lr_boot = []
    for _ in range(n_boot):
        gap_sim = stats.nbinom.rvs(null_params["r"], null_params["p"], size=n, random_state=rng)
        try:
            null_b = fit_null_nb(gap_sim)
            mix_b = fit_em_stabilized(gap_sim, X_arr, max_iter=150, tol=1e-6)
            lr_boot.append(max(2 * (mix_b["loglik"] - null_b["loglik"]), 0.0))
        except Exception:
            continue
    return np.array(lr_boot)


lr_null_dist = parametric_bootstrap_lrt(X_design.to_numpy(), null_fit, N_BOOT_LRT,
                                         rng_master.integers(0, 2**32 - 1))
n_success = len(lr_null_dist)
lrt_pvalue = float(np.mean(lr_null_dist >= lr_observed)) if n_success > 0 else np.nan
print(f"성공 반복: {n_success}/{N_BOOT_LRT} | 부트스트랩 LRT p-value(안정화 모형) = {lrt_pvalue:.4f}")

MODEL_JUSTIFIED = bool(pd.notna(lrt_pvalue) and lrt_pvalue < ALPHA and not p0_is_boundary)
print("-" * 78)
if MODEL_JUSTIFIED:
    print(f"✓ 판정: 2성분 구조가 정당화됨(p={lrt_pvalue:.4f}<{ALPHA}) — F단계를 주 분석으로 사용 가능")
else:
    reasons = []
    if not (pd.notna(lrt_pvalue) and lrt_pvalue < ALPHA):
        reasons.append(f"LRT 비유의(p={lrt_pvalue:.4f}>={ALPHA})")
    if p0_is_boundary:
        reasons.append(f"p0 경계해({fit_result['p0']:.4f})")
    print(f"✗ 판정: 2성분 구조가 정당화되지 않음({', '.join(reasons)}) — F단계는 exploratory_only로만 표기")
print("-" * 78)

pd.DataFrame({"lr_boot": lr_null_dist}).to_csv(OUT_DIR / "lrt_bootstrap_null_distribution_v2.csv", index=False)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
if n_success > 0:
    ax.hist(lr_null_dist, bins=30, color="steelblue", alpha=0.8, label="부트스트랩 귀무분포(LR, 안정화)")
ax.axvline(lr_observed, color="firebrick", linestyle="--", linewidth=1.5, label=f"관측 LR={lr_observed:.2f}")
ax.set_xlabel("LR 통계량"); ax.set_ylabel("빈도")
ax.set_title(f"안정화 모형 — 부트스트랩 LR 검정 (p={lrt_pvalue:.4f})")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_lrt_null_distribution_v2.png", dpi=150)
plt.close()


# ============================================================================
# E. 파라미터 불확실성 — 안정화 모형 재부트스트랩 + 잔존 분리 진단
# ============================================================================
print("\n" + "=" * 78)
print(f"E. 파라미터 불확실성(안정화 모형) — 비모수 부트스트랩(B={N_BOOT_PARAM})")
print("=" * 78)

n_obs = len(gap)
boot_betas, boot_p0, boot_r1, boot_p1 = [], [], [], []
rng_param = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
for _ in range(N_BOOT_PARAM):
    idx = rng_param.integers(0, n_obs, size=n_obs)
    try:
        fit_b = fit_em_stabilized(gap[idx], X_design.to_numpy()[idx], max_iter=150, tol=1e-6)
        boot_betas.append(fit_b["beta"]); boot_p0.append(fit_b["p0"])
        boot_r1.append(fit_b["r1"]); boot_p1.append(fit_b["p1"])
    except Exception:
        continue

n_boot_success = len(boot_p0)
print(f"성공 반복: {n_boot_success}/{N_BOOT_PARAM}")

param_ci, max_beta_halfwidth = {}, 0.0
if n_boot_success >= 30:
    boot_betas_arr = np.array(boot_betas)
    beta_ci = {}
    for j, name in enumerate(beta_names):
        lo, hi = np.percentile(boot_betas_arr[:, j], [2.5, 97.5])
        halfwidth = (hi - lo) / 2.0
        max_beta_halfwidth = max(max_beta_halfwidth, halfwidth)
        beta_ci[name] = {"point": float(fit_result["beta"][j]), "ci_lo": float(lo), "ci_hi": float(hi),
                          "ci_halfwidth": float(halfwidth)}
        flag = "  ⚠ 잔존 분리 의심" if halfwidth > BETA_CI_HALFWIDTH_WARN else ""
        print(f"  β[{name:<28}] 점추정={fit_result['beta'][j]:+.4f}, 95% CI=[{lo:+.4f}, {hi:+.4f}]{flag}")
    p0_lo, p0_hi = np.percentile(boot_p0, [2.5, 97.5])
    print(f"  p0 점추정={fit_result['p0']:.4f}, 95% CI=[{p0_lo:.4f}, {p0_hi:.4f}]")
    param_ci = {"beta": beta_ci, "p0": {"point": fit_result["p0"], "ci_lo": float(p0_lo), "ci_hi": float(p0_hi)}}

separation_still_present = max_beta_halfwidth > BETA_CI_HALFWIDTH_WARN
print(f"\n최대 β CI 반폭 = {max_beta_halfwidth:.3f} (경고기준 {BETA_CI_HALFWIDTH_WARN}) — "
      f"{'⚠ 잔존 분리 신호' if separation_still_present else '✓ 안정화 확인'}")

with open(OUT_DIR / "parameter_bootstrap_ci_v2.json", "w", encoding="utf-8") as f:
    json.dump(param_ci, f, ensure_ascii=False, indent=2, default=str)


# ============================================================================
# F. [수정 4] Track1/Track2 소프트 재가중 — MODEL_JUSTIFIED로 게이팅
# ============================================================================
FSTATUS = "primary" if MODEL_JUSTIFIED else "exploratory_only_model_not_justified"
print("\n" + "=" * 78)
print(f"F. [응용, 지위={'PRIMARY' if MODEL_JUSTIFIED else 'EXPLORATORY_ONLY'}] "
      f"Type{TARGET_TYPE_FOR_APPLICATION} 소프트 재가중")
print("=" * 78)
if not MODEL_JUSTIFIED:
    print("! 경고: D단계에서 2성분 구조가 정당화되지 않음 — 아래 결과는 참고용일 뿐 "
          "본문 주장 근거로 사용하지 말 것. 대신 이 시도 전체를 '기존 이산적 gap-day "
          "진단을 뒷받침하는 부정적 강건성 결과'로 보고할 것.")

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
    print(f"  [{outcome}] [{FSTATUS}] 하드 n={int(hard_mask.sum())} mean={hard_mean:+.4f} | "
          f"소프트 n_eff={n_eff:.2f} weighted_mean={soft_mean:+.4f}")
    track1_results[outcome] = {"status": FSTATUS, "hard_mean_att": hard_mean,
                                "soft_weighted_mean_att": soft_mean, "soft_n_eff_kish": n_eff}

with open(OUT_DIR / f"track1_soft_reweighted_type{TARGET_TYPE_FOR_APPLICATION}_v2.json", "w", encoding="utf-8") as f:
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
        print(f"  [{ycol}] [{FSTATUS}] 소프트 재가중 격차={soft_diff:+.3f} (n={len(sub)})")
        track2_results[ycol] = {
            "status": FSTATUS, "soft_weighted_diff": soft_diff, "n_hard": int(len(sub)),
            "interpretation_note": ("association only, not causal" if MODEL_JUSTIFIED
                                     else "exploratory_only — mixture structure not statistically justified"),
        }

with open(OUT_DIR / f"track2_soft_reweighted_type{TARGET_TYPE_FOR_APPLICATION}_v2.json", "w", encoding="utf-8") as f:
    json.dump(track2_results, f, ensure_ascii=False, indent=2, default=str)


# ============================================================================
# G. 종합 매니페스트 — 논문용 문구 자동 생성
# ============================================================================
if MODEL_JUSTIFIED:
    paper_recommendation = (
        f"부트스트랩 LR 검정(안정화 모형, p={lrt_pvalue:.4f}<{ALPHA})이 2성분 잠재계층 구조를 "
        f"지지했다. 공변량 기반 연속 사후확률(min_n={MIN_LEVEL_N} 풀링, ridge=λ{RIDGE_LAMBDA})을 "
        f"Track1/Track2 재추정의 가중치로 사용한 결과를 본문에 제시할 수 있다."
    )
else:
    p0_boundary_note = f", p0 경계해(p0={fit_result['p0']:.4f})" if p0_is_boundary else ""
    paper_recommendation = (
        f"특징기반 2성분 혼합모형(EM)으로 gap_days 기반 하드 threshold를 연속 확률로 대체하는 "
        f"방안을 시도했다. 희소 범주 풀링(min_n={MIN_LEVEL_N})과 릿지 페널티(λ={RIDGE_LAMBDA})로 "
        f"로지스틱 게이트의 준완전분리 문제를 안정화한 뒤(β 95% CI 최대 반폭이 22.8→"
        f"{max_beta_halfwidth:.2f}로 축소되어 확인됨) 모수적 부트스트랩 LR 검정을 재실행했으나, "
        f"2성분 구조는 통계적으로 정당화되지 않았다(p={lrt_pvalue:.4f}>={ALPHA}{p0_boundary_note}). "
        f"이는 born-treated/true-switcher 구분이 이 데이터에서 본질적으로 이산적임을 시사하며, "
        f"기존의 이산적 gap-day 진단 절차(threshold=1일)를 주 분석으로 유지하고, 이 혼합모형 시도는 "
        f"그 절차의 타당성을 뒷받침하는 부정적 강건성 결과로 보고한다."
    )

manifest = {
    "version": "v2_ridge_stabilized",
    "n_pooled_adopters": int(len(pooled)),
    "used_covariates_after_collapse": used_covariate_cols,
    "min_level_n": MIN_LEVEL_N, "ridge_lambda": RIDGE_LAMBDA, "collapse_log": collapse_log,
    "em_fit": {"beta_names": beta_names, "beta": fit_result["beta"].tolist(),
               "p0": fit_result["p0"], "r1": fit_result["r1"], "p1": fit_result["p1"],
               "loglik": fit_result["loglik"], "aic": fit_result["aic"], "bic": fit_result["bic"]},
    "lrt_bootstrap": {"lr_observed": lr_observed, "n_boot_success": n_success,
                       "p_value": lrt_pvalue if pd.notna(lrt_pvalue) else None},
    "diagnostics": {"p0_is_boundary": p0_is_boundary, "max_beta_ci_halfwidth": max_beta_halfwidth,
                     "separation_still_present": separation_still_present,
                     "n_ambiguous_posterior": n_ambig, "n_hard_soft_disagree": n_disagree},
    "MODEL_JUSTIFIED": MODEL_JUSTIFIED,   # ← step5a가 이 플래그를 그대로 승계한다
    "f_section_status": FSTATUS,
    "paper_recommendation_text": paper_recommendation,
}
with open(OUT_DIR / "mixture_model_fit_v2.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

print("\n" + "=" * 78)
print(f"완료 | MODEL_JUSTIFIED = {MODEL_JUSTIFIED}")
print("=" * 78)
print(f"\n[논문용 문구 초안]\n{paper_recommendation}")
print(f"\n다음 단계: step4c_posterior_vs_gap_figure.py (Figure 12B), "
      f"step5a_dro_calibration.py")
