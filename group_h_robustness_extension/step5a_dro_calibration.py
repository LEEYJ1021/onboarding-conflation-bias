"""
================================================================================
step5a_dro_calibration.py
DRO Worst-Case Bound — [A] Track2 원자료 재구성 + [B/C] KL 앙상블 반경 캘리브레이션
================================================================================
step5b(worst-case bound grid)와 step5c(breakdown+placebo)가 공유해서 쓸
"born-treated vs 매칭대조군" 개별 고객 단위 원자료와, DRO 앙상블 반경(eps)의
캘리브레이션 기준을 이 스크립트 하나에서 만들어 파일로 넘긴다.

설계 원칙 (① censored-mixture 결과와의 연결):
    step4b(v2)의 결론은 MODEL_JUSTIFIED=False(LRT p=0.22)였다. 따라서 ①의
    사후확률(posterior_born_treated_prob)을 이 DRO 분석의 **주 캘리브레이션**
    으로 쓰면 정당화되지 않은 모형 위에 또 다른 모형을 쌓는 꼴이 된다. 대신:
      - [주 캘리브레이션] step3g에서 이미 검증된, outcome과 시간적으로 겹치지
        않는 공변량(log_total_cost_excl_window, device_type_mode)으로 표준
        (비혼합) 로지스틱 성향점수 모형을 적합하고, IPW 재가중에 필요한 KL
        발산량을 eps_calibrated_main으로 채택한다.
      - [보조/탐색적] ①의 사후확률 기반 대안 eps도 계산해 참고용으로만 남기고,
        MODEL_JUSTIFIED 플래그를 그대로 승계해 매니페스트에 기록한다.

출력 (dro_output_v4/):
    track2_raw_data_type{T}.csv   — 개별 고객 단위 원자료(outcome, 공변량,
                                     is_born_treated, propensity_score)
                                     → step5b/step5c가 그대로 읽는다
    dro_calibration_manifest.json — eps_calibrated_main, 성향점수 모형 계수,
                                     공변량 균형 체크, MODEL_JUSTIFIED 승계
================================================================================
"""
import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================================
# 0. 경로 및 설정
# ============================================================================
ROOT = Path(os.environ.get("AD_DATA_ROOT", "./master_dataset"))
STEP2_DIR = Path(os.environ.get("STEP2_OUT", str(ROOT / "step2_treatment_output")))
MIXTURE_V2_DIR = Path(os.environ.get("CENSORED_OUT", str(STEP2_DIR / "censored_mixture_output_v2")))
OUT_DIR = Path(os.environ.get("DRO_OUT", str(STEP2_DIR / "dro_output_v4")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TYPE = int(os.environ.get("TARGET_TYPE_APPLICATION", "6"))
GAP_THRESHOLD = int(os.environ.get("BORN_TREATED_GAP_THRESHOLD", "1"))
EARLY_WINDOW = int(os.environ.get("EARLY_OUTCOME_WINDOW_DAYS", "30"))
REG_WINDOW_MAIN = int(os.environ.get("REG_MATCH_WINDOW_DAYS", "14"))
OUTCOME_VARS = ["log_spend_safe", "n_campaign_types_active"]

PANEL_PATH = ROOT / "customer_day_panel.csv"
CTP_PATH = ROOT / "customer_day_campaign_type_panel.csv"
MASTER_PATH = ROOT / "df_analysis_master.csv"
FINAL_COHORT_PATH = STEP2_DIR / f"staggered_adoption_FINAL_type{TARGET_TYPE}.csv"
BORN_DIAG_PATH = STEP2_DIR / f"step3c_born_treated_diagnosis_type{TARGET_TYPE}.csv"
MIXTURE_MANIFEST_PATH = MIXTURE_V2_DIR / "mixture_model_fit_v2.json"
MIXTURE_FEATURES_PATH = MIXTURE_V2_DIR / "hard_vs_soft_classification_v2.csv"
MIXTURE_POSTERIOR_COL = "posterior_born_treated_prob"

for p in [PANEL_PATH, CTP_PATH, MASTER_PATH, FINAL_COHORT_PATH, BORN_DIAG_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"필요한 입력 파일이 없습니다: {p}")


def _detect_date_column(df: pd.DataFrame) -> str:
    for c in ("date", "stat_date", "stat_dt", "dt", "ad_date", "report_date", "log_date"):
        if c in df.columns:
            return c
    for c in df.columns:
        try:
            pd.to_datetime(df[c].dropna().iloc[:5])
            return c
        except (ValueError, TypeError):
            continue
    raise KeyError("날짜 컬럼 자동감지 실패")


# ============================================================================
# A. 원자료 재구성 — 개별 대조군/born-treated 고객 단위 outcome 값
# ============================================================================
print("=" * 78)
print("A. Track2 원자료 재구성 — born-treated 및 매칭대조군 개별 outcome")
print("=" * 78)

panel = pd.read_csv(PANEL_PATH)
ctp = pd.read_csv(CTP_PATH)
master = pd.read_csv(MASTER_PATH)
final_cohort = pd.read_csv(FINAL_COHORT_PATH)
born_diag = pd.read_csv(BORN_DIAG_PATH)

DATE_COL = _detect_date_column(panel)
DATE_COL_CTP = _detect_date_column(ctp)
panel[DATE_COL] = pd.to_datetime(panel[DATE_COL])
ctp[DATE_COL_CTP] = pd.to_datetime(ctp[DATE_COL_CTP])
ctp["campaign_type"] = pd.to_numeric(ctp["campaign_type"], errors="coerce")
master["date"] = pd.to_datetime(master["date"])
final_cohort["final_first_treated_date"] = pd.to_datetime(final_cohort["final_first_treated_date"])
born_diag["raw_panel_date_min"] = pd.to_datetime(born_diag["raw_panel_date_min"])

panel_ext = panel[["customer_id", DATE_COL, "cost"]].rename(columns={DATE_COL: "date"}).copy()
panel_ext["log_spend_safe"] = np.log1p(panel_ext["cost"])

ctp_active = ctp[ctp["cost"] > 0].copy()
n_types_active = (
    ctp_active.groupby(["customer_id", DATE_COL_CTP])["campaign_type"].nunique()
    .rename("n_campaign_types_active").reset_index().rename(columns={DATE_COL_CTP: "date"})
)
panel_ext = panel_ext.merge(n_types_active, on=["customer_id", "date"], how="left")
panel_ext["n_campaign_types_active"] = panel_ext["n_campaign_types_active"].fillna(0).astype(int)

master_outcomes = master[["customer_id", "date"] + OUTCOME_VARS].copy()
master_outcomes["_source"] = "stable_window(master)"
panel_ext_labeled = panel_ext[["customer_id", "date"] + OUTCOME_VARS].copy()
panel_ext_labeled["_source"] = "raw_panel(pre_or_post_window)"
combined_outcomes = pd.concat([master_outcomes, panel_ext_labeled], ignore_index=True)
combined_outcomes = (
    combined_outcomes.sort_values(
        "_source", key=lambda s: s.map({"stable_window(master)": 0, "raw_panel(pre_or_post_window)": 1}))
    .drop_duplicates(subset=["customer_id", "date"], keep="first").drop(columns="_source")
)

never_treated_ids = final_cohort.loc[final_cohort["final_cohort"] == "never_treated", "customer_id"].unique().tolist()
never_treated_reg = (panel[panel["customer_id"].isin(never_treated_ids)]
                      .groupby("customer_id")[DATE_COL].min().rename("registration_date").reset_index())

born_treated = born_diag[born_diag["gap_days"] <= GAP_THRESHOLD][["customer_id", "raw_panel_date_min"]] \
    .rename(columns={"raw_panel_date_min": "registration_date"}).copy()


def early_outcome(cid, reg_date, ycol):
    sub = combined_outcomes[(combined_outcomes["customer_id"] == cid)
                             & (combined_outcomes["date"] >= reg_date)
                             & (combined_outcomes["date"] < reg_date + pd.Timedelta(days=EARLY_WINDOW))]
    if sub.empty:
        return np.nan
    return float(sub[ycol].sum()) if ycol == "log_spend_safe" else float(sub[ycol].mean())


matched_control_ids = set()
for _, bt in born_treated.iterrows():
    reg_bt = bt["registration_date"]
    candidates = never_treated_reg[
        (never_treated_reg["registration_date"] >= reg_bt - pd.Timedelta(days=REG_WINDOW_MAIN))
        & (never_treated_reg["registration_date"] <= reg_bt + pd.Timedelta(days=REG_WINDOW_MAIN))]
    matched_control_ids.update(candidates["customer_id"].tolist())

print(f"born-treated: {len(born_treated)}명, 매칭대조군(중복제거): {len(matched_control_ids)}명")

rows = [{"customer_id": bt["customer_id"], "registration_date": bt["registration_date"], "is_born_treated": 1}
        for _, bt in born_treated.iterrows()]
for cid in sorted(matched_control_ids):
    reg = never_treated_reg.loc[never_treated_reg["customer_id"] == cid, "registration_date"].iloc[0]
    rows.append({"customer_id": cid, "registration_date": reg, "is_born_treated": 0})
raw_df = pd.DataFrame(rows)
for ycol in OUTCOME_VARS:
    raw_df[ycol] = raw_df.apply(lambda r: early_outcome(r["customer_id"], r["registration_date"], ycol), axis=1)


# ============================================================================
# B. [주 캘리브레이션] step3g의 outcome-비오염 공변량으로 표준 성향점수 모형
# ============================================================================
print("\n" + "=" * 78)
print("B. [주 캘리브레이션] 표준 로지스틱 성향점수 모형 (outcome-비오염 공변량)")
print("=" * 78)


def window_cost(cid, reg_date):
    sub = panel[(panel["customer_id"] == cid) & (panel[DATE_COL] >= reg_date)
                & (panel[DATE_COL] < reg_date + pd.Timedelta(days=EARLY_WINDOW))]
    return float(sub["cost"].sum())


total_cost_full = panel.groupby("customer_id")["cost"].sum().rename("total_cost_full_history")
raw_df = raw_df.merge(total_cost_full, on="customer_id", how="left")
raw_df["total_cost_full_history"] = raw_df["total_cost_full_history"].fillna(0.0)
raw_df["window_cost_raw"] = raw_df.apply(lambda r: window_cost(r["customer_id"], r["registration_date"]), axis=1)
raw_df["total_cost_excl_window"] = (raw_df["total_cost_full_history"] - raw_df["window_cost_raw"]).clip(lower=0.0)
raw_df["log_total_cost_excl_window"] = np.log1p(raw_df["total_cost_excl_window"])

level_cols = ["customer_id"] + [c for c in ["device_type_mode"] if c in master.columns]
raw_df = raw_df.merge(master.drop_duplicates("customer_id")[level_cols], on="customer_id", how="left")
if "device_type_mode" not in raw_df.columns:
    raw_df["device_type_mode"] = "unknown"
raw_df["device_type_mode"] = raw_df["device_type_mode"].fillna("unknown")

bt_cov = raw_df.loc[raw_df["is_born_treated"] == 1, "log_total_cost_excl_window"].dropna()
ct_cov = raw_df.loc[raw_df["is_born_treated"] == 0, "log_total_cost_excl_window"].dropna()
t_bal, p_bal = stats.ttest_ind(bt_cov, ct_cov, equal_var=False)
print(f"공변량 불균형(재확인): log_total_cost_excl_window Welch t={t_bal:.3f}, p={p_bal:.4f}")

X_cols_dummy = pd.get_dummies(raw_df[["device_type_mode"]].astype(str), drop_first=True)
X_ps = pd.concat([raw_df[["log_total_cost_excl_window"]].fillna(raw_df["log_total_cost_excl_window"].median()),
                   X_cols_dummy], axis=1)
X_ps = sm.add_constant(X_ps, has_constant="add").astype(float)
y_ps = raw_df["is_born_treated"].to_numpy()

ps_model = sm.GLM(y_ps, X_ps, family=sm.families.Binomial()).fit()
raw_df["propensity_score"] = ps_model.predict(X_ps)
print("\n성향점수 모형 계수:")
print(ps_model.params.round(4).to_string())

ctrl_mask = raw_df["is_born_treated"] == 0
ps_ctrl = raw_df.loc[ctrl_mask, "propensity_score"].clip(1e-4, 1 - 1e-4).to_numpy()
ipw_raw = ps_ctrl / (1 - ps_ctrl)
ipw_norm = ipw_raw / ipw_raw.sum()
n_ctrl = len(ipw_norm)

ctrl_cov_vals = raw_df.loc[ctrl_mask, "log_total_cost_excl_window"].to_numpy()
mean_ctrl_unweighted = float(np.mean(ctrl_cov_vals))
mean_ctrl_ipw = float(np.sum(ipw_norm * ctrl_cov_vals))
mean_bt_cov = float(bt_cov.mean())
print(f"\nIPW 재가중 sanity check (log_total_cost_excl_window):")
print(f"  born-treated 평균         = {mean_bt_cov:.3f}")
print(f"  대조군 평균(재가중 전)     = {mean_ctrl_unweighted:.3f}  (격차 {mean_bt_cov - mean_ctrl_unweighted:+.3f})")
print(f"  대조군 평균(IPW 재가중 후) = {mean_ctrl_ipw:.3f}  (격차 {mean_bt_cov - mean_ctrl_ipw:+.3f})")
balance_improved = abs(mean_bt_cov - mean_ctrl_ipw) < abs(mean_bt_cov - mean_ctrl_unweighted)
print(f"  {'✓ IPW 재가중이 공변량 격차를 실제로 줄임 — 캘리브레이션 타당' if balance_improved else '⚠ IPW 재가중이 격차를 줄이지 못함 — 성향점수 모형 재검토 필요'}")

kl_calibrated_main = float(np.sum(ipw_norm * np.log(n_ctrl * ipw_norm + 1e-12)))
print(f"\n[주 캘리브레이션] KL(IPW재가중분포 || 균등분포) = eps_calibrated_main = {kl_calibrated_main:.4f}")


# ============================================================================
# C. [보조/탐색적 캘리브레이션] ①(v2) 사후확률 기반 — MODEL_JUSTIFIED 플래그 승계
# ============================================================================
print("\n" + "=" * 78)
print("C. [보조/탐색적] ① v2 사후확률 기반 대안 eps (참고용)")
print("=" * 78)

mixture_model_justified = None
kl_calibrated_aux = None

if MIXTURE_MANIFEST_PATH.exists():
    with open(MIXTURE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        mixture_manifest = json.load(f)
    mixture_model_justified = bool(mixture_manifest.get("MODEL_JUSTIFIED", False))
    print(f"① v2 매니페스트 로드: MODEL_JUSTIFIED = {mixture_model_justified}")
else:
    print(f"① v2 매니페스트({MIXTURE_MANIFEST_PATH})가 없습니다 — 없이 진행")

if MIXTURE_FEATURES_PATH.exists():
    mixture_feat = pd.read_csv(MIXTURE_FEATURES_PATH)
    if {"customer_id", "campaign_type", MIXTURE_POSTERIOR_COL}.issubset(mixture_feat.columns):
        posterior_map = mixture_feat.loc[
            mixture_feat["campaign_type"] == TARGET_TYPE, ["customer_id", MIXTURE_POSTERIOR_COL]
        ].set_index("customer_id")
        matched_posterior = posterior_map.reindex(
            raw_df.loc[raw_df["is_born_treated"] == 1, "customer_id"]).dropna()
        if len(matched_posterior) >= 5:
            kl_calibrated_aux = float((1 - matched_posterior[MIXTURE_POSTERIOR_COL]).mean())
            print(f"  born-treated {len(matched_posterior)}명의 평균 (1-사후확률) = {kl_calibrated_aux:.4f} (참고용)")
else:
    print(f"① v2 사후확률 파일({MIXTURE_FEATURES_PATH})이 없습니다 — 보조 캘리브레이션 생략.")

if mixture_model_justified is False:
    print("\n⚠ ①의 결론(MODEL_JUSTIFIED=False)에 따라, ① 사후확률은 이번 DRO 분석의 "
          "주 캘리브레이션으로 사용하지 않는다. [B]의 표준 성향점수 캘리브레이션만 채택.")


# ============================================================================
# D. 산출물 저장
# ============================================================================
raw_data_path = OUT_DIR / f"track2_raw_data_type{TARGET_TYPE}.csv"
raw_df.to_csv(raw_data_path, index=False)

calibration_manifest = {
    "target_type": TARGET_TYPE,
    "reg_match_window_days": REG_WINDOW_MAIN,
    "early_outcome_window_days": EARLY_WINDOW,
    "outcome_vars": OUTCOME_VARS,
    "n_born_treated": int((raw_df["is_born_treated"] == 1).sum()),
    "n_matched_controls": int((raw_df["is_born_treated"] == 0).sum()),
    "propensity_model_coefficients": ps_model.params.to_dict(),
    "covariate_balance_check": {
        "mean_born_treated": mean_bt_cov, "mean_control_unweighted": mean_ctrl_unweighted,
        "mean_control_ipw_reweighted": mean_ctrl_ipw, "welch_t_pvalue": float(p_bal),
        "ipw_improves_balance": balance_improved,
    },
    "eps_calibrated_main": kl_calibrated_main,
    "eps_calibrated_auxiliary_from_mixture_v2": kl_calibrated_aux,
    "mixture_v1v2_model_justified": mixture_model_justified,
    "raw_data_path": str(raw_data_path),
}
manifest_path = OUT_DIR / "dro_calibration_manifest.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(calibration_manifest, f, ensure_ascii=False, indent=2, default=str)

print("\n" + "=" * 78)
print("완료")
print("=" * 78)
print(f"원자료 저장: {raw_data_path}")
print(f"캘리브레이션 매니페스트 저장: {manifest_path}")
print(f"\n다음 단계: step5b_dro_worst_case_bound.py, step5c_dro_breakdown_and_placebo_v4.py")
