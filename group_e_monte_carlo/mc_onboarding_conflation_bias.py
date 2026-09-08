"""
================================================================================
Monte Carlo Validation: "Onboarding Conflation Bias" and the Gap-Day
Diagnostic Protocol (Track 1 / Track 2 split)
================================================================================
Purpose:
    The empirical finding — that ~81% of nominally "adopting" customers
    across four campaign types are actually born-treated (panel-entry
    date == adoption date) — motivates two downstream analysis tracks:
      Track 1 (within-customer DiD): born-treated customers have no
        pre-period by construction, so they are automatically dropped.
        Point estimates should stay unbiased, but nominal N overstates
        the identifying sample, so naive (nominal-N) standard errors
        understate uncertainty.
      Track 2 (cross-sectional matched comparison): born-treated
        customers' early-window outcomes are contaminated by a novelty/
        onboarding effect that is indistinguishable from a real
        treatment effect in a cross-sectional design, so even a zero
        true effect can appear strongly "significant" at large N.

    This simulation generates synthetic panels with a known true effect
    and a known novelty effect, varies the born-treated share p_born from
    0 to 1, and quantifies both mechanisms.

Design note (bug fix retained from development log): pre-period windows
    must not overlap the novelty window for true switchers, or Track 1
    bias is contaminated regardless of p_born. GAP_MIN is set well above
    EVENT_WINDOW + NOVELTY_DURATION to guarantee this.

Output: mc_output/mc_summary_by_p_born.csv / .json
        mc_output/mc_bias_vs_pborn.png
        mc_output/mc_effective_n_vs_pborn.png
        mc_output/mc_se_overconfidence_vs_pborn.png
================================================================================
"""
import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="Mean of empty slice")

OUT_DIR = Path(os.environ.get("MC_OUT", "./mc_output"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = int(os.environ.get("MC_SEED", "20260908"))
rng_master = np.random.default_rng(SEED)

N_ADOPTERS = int(os.environ.get("N_ADOPTERS", "35"))   # calibrated to empirical robust cohort
N_NEVER = int(os.environ.get("N_NEVER", "61"))
EVENT_WINDOW = int(os.environ.get("EVENT_WINDOW_DAYS", "30"))
NOVELTY_DURATION = int(os.environ.get("NOVELTY_DURATION_DAYS", "10"))

_MIN_SAFE_GAP = EVENT_WINDOW + NOVELTY_DURATION
GAP_MIN = int(os.environ.get("GAP_MIN_DAYS", str(_MIN_SAFE_GAP + 5)))
GAP_MAX = int(os.environ.get("GAP_MAX_DAYS", str(_MIN_SAFE_GAP + 50)))
assert GAP_MIN >= _MIN_SAFE_GAP, "GAP_MIN too small: true-switcher pre-period would overlap novelty window."

NOVELTY_EFFECT = float(os.environ.get("NOVELTY_EFFECT", "0.6"))
TRUE_EFFECT = float(os.environ.get("TRUE_EFFECT", "0.15"))
SIGMA_MU, SIGMA_EPS = 0.8, 0.5
N_REPS = int(os.environ.get("MC_REPS", "2000"))
P_BORN_GRID = [0.0, 0.2, 0.4, 0.6, 0.74, 0.8, 0.9, 1.0]  # 0.74-0.80 matches empirical range

print(f"N_ADOPTERS={N_ADOPTERS}, N_NEVER={N_NEVER}, EVENT_WINDOW={EVENT_WINDOW}d, "
      f"NOVELTY_DURATION={NOVELTY_DURATION}d, safe gap>={_MIN_SAFE_GAP}d")
print(f"TRUE_EFFECT={TRUE_EFFECT}, NOVELTY_EFFECT={NOVELTY_EFFECT}, N_REPS={N_REPS}\n")


def simulate_one_rep(p_born: float, rng: np.random.Generator):
    is_born = rng.random(N_ADOPTERS) < p_born
    gap = np.where(is_born, 0, rng.integers(GAP_MIN, GAP_MAX + 1, size=N_ADOPTERS))
    mu_adopt = rng.normal(0.0, SIGMA_MU, size=N_ADOPTERS)

    k = np.arange(1, EVENT_WINDOW + 1)
    day_since_reg_pre = gap[:, None] - k[None, :]
    valid_pre_mask = day_since_reg_pre >= 0
    novelty_pre = np.where(day_since_reg_pre < NOVELTY_DURATION, NOVELTY_EFFECT, 0.0)
    y_pre = mu_adopt[:, None] + novelty_pre + rng.normal(0.0, SIGMA_EPS, size=(N_ADOPTERS, EVENT_WINDOW))
    y_pre = np.where(valid_pre_mask, y_pre, np.nan)
    pre_mean, pre_n = np.nanmean(y_pre, axis=1), valid_pre_mask.sum(axis=1)

    j = np.arange(0, EVENT_WINDOW)
    day_since_reg_post = gap[:, None] + j[None, :]
    novelty_post = np.where(day_since_reg_post < NOVELTY_DURATION, NOVELTY_EFFECT, 0.0)
    y_post = mu_adopt[:, None] + TRUE_EFFECT + novelty_post + rng.normal(0.0, SIGMA_EPS, size=(N_ADOPTERS, EVENT_WINDOW))
    post_mean = y_post.mean(axis=1)

    mu_never = rng.normal(0.0, SIGMA_MU, size=N_NEVER)
    ctrl_pre = mu_never[:, None] + rng.normal(0.0, SIGMA_EPS, size=(N_NEVER, EVENT_WINDOW))
    ctrl_post = mu_never[:, None] + rng.normal(0.0, SIGMA_EPS, size=(N_NEVER, EVENT_WINDOW))
    ctrl_pre_grand, ctrl_post_grand = ctrl_pre.mean(axis=1).mean(), ctrl_post.mean(axis=1).mean()

    att_i = (post_mean - pre_mean) - (ctrl_post_grand - ctrl_pre_grand)
    valid = pre_n > 0
    n_switch = int(valid.sum())
    if n_switch > 1:
        att_valid = att_i[valid]
        naive_pooled_att = float(np.mean(att_valid))
        att_std = float(np.std(att_valid, ddof=1))
        se_correct = att_std / np.sqrt(n_switch)
        se_naive_nominal = att_std / np.sqrt(N_ADOPTERS)
    else:
        naive_pooled_att = se_correct = se_naive_nominal = np.nan

    born_post = post_mean[is_born]
    if len(born_post) >= 3:
        track2_est = float(born_post.mean() - ctrl_post_grand)
        _, pval2 = sps.ttest_ind(born_post, ctrl_post.mean(axis=1), equal_var=False)
    else:
        track2_est, pval2 = np.nan, np.nan

    return dict(n_switch=n_switch, naive_pooled_att=naive_pooled_att, se_correct=se_correct,
                se_naive_nominal=se_naive_nominal, track2_est=track2_est, track2_pvalue=pval2)


all_rows = []
for p_born in P_BORN_GRID:
    rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
    rep_df = pd.DataFrame([simulate_one_rep(p_born, rng) for _ in range(N_REPS)])
    rep_df["p_born"] = p_born
    all_rows.append(rep_df)

full_reps_df = pd.concat(all_rows, ignore_index=True)

summary_rows = []
for p_born, g in full_reps_df.groupby("p_born"):
    t1, t2 = g["naive_pooled_att"].dropna(), g["track2_est"].dropna()
    se_correct_avg, se_naive_avg = g["se_correct"].mean(), g["se_naive_nominal"].mean()
    summary_rows.append({
        "p_born": p_born, "avg_n_switch": g["n_switch"].mean(), "n_nominal": N_ADOPTERS,
        "track1_bias": (t1.mean() - TRUE_EFFECT) if len(t1) else np.nan,
        "track1_empirical_sd": t1.std(ddof=1) if len(t1) > 1 else np.nan,
        "track1_se_correct_avg": se_correct_avg, "track1_se_naive_nominal_avg": se_naive_avg,
        "track1_se_understatement_pct": (1 - se_naive_avg / se_correct_avg) * 100
            if pd.notna(se_correct_avg) and se_correct_avg > 0 else np.nan,
        "track2_bias": (t2.mean() - TRUE_EFFECT) if len(t2) else np.nan,
        "track2_pct_significant": (g["track2_pvalue"] < 0.05).mean() if g["track2_pvalue"].notna().any() else np.nan,
    })
summary_df = pd.DataFrame(summary_rows).sort_values("p_born").reset_index(drop=True)
summary_df.to_csv(OUT_DIR / "mc_summary_by_p_born.csv", index=False)
print(summary_df.round(4).to_string(index=False))

# --- Sanity check: at p_born=0, Track 1 bias should be ~0 ---
row0 = summary_df[np.isclose(summary_df["p_born"], 0.0)].iloc[0]
se_of_mean_at_0 = row0["track1_empirical_sd"] / np.sqrt(N_REPS) if pd.notna(row0["track1_empirical_sd"]) else np.nan
sanity_passed = bool(pd.notna(se_of_mean_at_0) and abs(row0["track1_bias"]) <= 3 * se_of_mean_at_0)
print(f"\nSanity check (p_born=0, Track1 bias should ~ 0): bias={row0['track1_bias']:+.4f}, "
      f"MC SE={se_of_mean_at_0:.4f} -> {'PASS' if sanity_passed else 'FAIL'}")

# --- Closed-form theoretical value for Track 2 bias ---
theoretical_track2_bias = NOVELTY_EFFECT * (NOVELTY_DURATION / EVENT_WINDOW)
nonzero = summary_df[summary_df["p_born"] > 0]
obs_range = (nonzero["track2_bias"].min(), nonzero["track2_bias"].max()) if not nonzero.empty else (np.nan, np.nan)
print(f"Track2 theoretical bias = NOVELTY_EFFECT x (NOVELTY_DURATION/EVENT_WINDOW) = {theoretical_track2_bias:.4f}")
print(f"Simulated range (p_born>0): [{obs_range[0]:.4f}, {obs_range[1]:.4f}] -> "
      f"{'matches' if obs_range[0]-0.02 <= theoretical_track2_bias <= obs_range[1]+0.02 else 'MISMATCH'}")

# --- Plots ---
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.axhline(0, color="gray", linestyle=":", linewidth=1)
ax.plot(summary_df["p_born"], summary_df["track1_bias"], marker="o", color="steelblue", label="Track 1 (within-customer DiD)")
ax.plot(summary_df["p_born"], summary_df["track2_bias"], marker="s", color="firebrick", label="Track 2 (cross-sectional matching)")
ax.axvspan(0.70, 0.85, color="gold", alpha=0.15, label="empirical range (74-81%)")
ax.set_xlabel("Born-treated share (p_born)"); ax.set_ylabel("Bias (estimate - true effect)")
ax.set_title(f"Track 1 vs Track 2 Bias (true effect={TRUE_EFFECT}, novelty effect={NOVELTY_EFFECT})")
ax.legend(fontsize=8); plt.tight_layout()
plt.savefig(OUT_DIR / "mc_bias_vs_pborn.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.plot(summary_df["p_born"], summary_df["avg_n_switch"], marker="o", color="darkgreen", label="Effective N")
ax.axhline(N_ADOPTERS, color="gray", linestyle="--", label=f"Nominal N ({N_ADOPTERS})")
ax.axvspan(0.70, 0.85, color="gold", alpha=0.15, label="empirical range (74-81%)")
ax.set_xlabel("Born-treated share (p_born)"); ax.set_ylabel("Number of adopters")
ax.set_title("Nominal vs. Effective Identifying Sample (Track 1)")
ax.legend(fontsize=8); plt.tight_layout()
plt.savefig(OUT_DIR / "mc_effective_n_vs_pborn.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.plot(summary_df["p_born"], summary_df["track1_se_correct_avg"], marker="o", color="steelblue", label="Correct SE (effective N)")
ax.plot(summary_df["p_born"], summary_df["track1_se_naive_nominal_avg"], marker="s", color="orange", label=f"Naive SE (nominal N={N_ADOPTERS})")
ax.axvspan(0.70, 0.85, color="gold", alpha=0.15, label="empirical range (74-81%)")
ax.set_xlabel("Born-treated share (p_born)"); ax.set_ylabel("Standard error")
ax.set_title("SE Understatement from Using Nominal N")
ax.legend(fontsize=8); plt.tight_layout()
plt.savefig(OUT_DIR / "mc_se_overconfidence_vs_pborn.png", dpi=150); plt.close()

with open(OUT_DIR / "mc_summary_by_p_born.json", "w") as f:
    json.dump({
        "config": dict(N_ADOPTERS=N_ADOPTERS, N_NEVER=N_NEVER, EVENT_WINDOW=EVENT_WINDOW,
                        GAP_MIN=GAP_MIN, GAP_MAX=GAP_MAX, NOVELTY_DURATION=NOVELTY_DURATION,
                        NOVELTY_EFFECT=NOVELTY_EFFECT, TRUE_EFFECT=TRUE_EFFECT, N_REPS=N_REPS),
        "sanity_check_p_born_0": {"track1_bias": float(row0["track1_bias"]), "passed": sanity_passed},
        "track2_theoretical_vs_simulated": {"theoretical_bias": float(theoretical_track2_bias),
                                             "simulated_range": [float(obs_range[0]), float(obs_range[1])]},
        "summary_by_p_born": summary_df.to_dict("records"),
    }, f, indent=2, default=str)
print(f"\nSaved to {OUT_DIR}/: mc_summary_by_p_born.csv/.json, 3 PNG figures")
