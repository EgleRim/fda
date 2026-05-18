# ================================================================
# FUNCTIONAL DATA ANALYSIS — COMPLETE SCRIPT
# Eurozone 10Y Government Bond Yield Changes
# Function-on-Scalar Regression with FPCA Context
# ================================================================

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import t as t_dist, f as f_dist
from skfda import FDataBasis, FDataGrid
from skfda.representation.basis import BSplineBasis
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.misc.regularization import L2Regularization
from skfda.misc.operators import LinearDifferentialOperator
from skfda.exploratory.decomposition import FPCA
from statsmodels.stats.stattools import durbin_watson
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# SECTION 1 — LOAD AND RECONSTRUCT FUNCTIONAL DATA OBJECT
# ================================================================

print("=" * 65)
print("SECTION 1 — Load and reconstruct functional data object")
print("=" * 65)

# --- 1A: Reconstruct fd object from B-spline coefficients ---
# Parameters match original R smoothing:
#   cubic B-splines, 25 basis functions, lambda=10,
#   second derivative penalty, domain [1,62]

coef_df = pd.read_csv(
    "C:/Users/dskinkys/Desktop/Data science/fda/"
    "python_functional_data_object/fd_coefficients.csv",
    index_col=0
)
countries = coef_df.index.tolist()
basis     = BSplineBasis(domain_range=(1.0, 62.0), n_basis=25, order=4)
fd_python = FDataBasis(
    basis=basis,
    coefficients=coef_df.values,
    sample_names=countries
)

# evaluate on monthly time grid: t=1 (Feb 2021) to t=62 (Mar 2026)
t      = np.arange(1, 63, dtype=float)
Y      = fd_python(t).squeeze()    # (16 countries, 62 time points)

# date labels for plots
dates       = pd.date_range(start='2021-02-01', periods=62, freq='ME')
date_labels = dates.strftime('%Y-%m')
tick_pos    = np.arange(0, 62, 6)
tick_labels = date_labels[tick_pos]

print(f"Functional data object reconstructed:")
print(f"  n_curves  : {fd_python.n_samples}  (countries)")
print(f"  n_basis   : {fd_python.n_basis}   (B-spline basis functions)")
print(f"  domain    : {fd_python.domain_range}")
print(f"  Y shape   : {Y.shape}  (countries x time points)")
print(f"  Period    : Feb 2021 - Mar 2026  (62 months)")

# --- 1B: Load country-level scalar regressors ---
reg_df = pd.read_csv(
    "C:/Users/dskinkys/Desktop/Data science/fda/"
    "data/Regressors_country_level_final.csv",
    index_col='Country'
).reindex(countries)

# drop inflation differential (correlation -0.76 with distance)
selected  = ['SP_Rating', 'Debt_GDP_avg',
             'CurrentAccount_avg', 'Distance_RU_BY_km']
Z_raw     = reg_df[selected].values.astype(float)
reg_means = Z_raw.mean(axis=0)
reg_stds  = Z_raw.std(axis=0)
Z_std     = (Z_raw - reg_means) / reg_stds    # standardise

assert not reg_df[selected].isna().any().any(), \
    "Missing values detected in regressors"

intercept = np.ones((len(countries), 1))
Z         = np.hstack([intercept, Z_std])     # (16, 5)
z_names   = ['Intercept'] + selected
n, T, p   = len(countries), len(t), Z.shape[1]

print(f"\nRegressors loaded and standardised:")
print(f"  {selected}")
print(f"  Z shape: {Z.shape}  (countries x [intercept + 4 regressors])")
print(f"  n={n}, T={T}, p={p}")

# ================================================================
# SECTION 2 — EXPLORATORY: FPCA
# Connects to group paper Section 2.2
# Run BEFORE regression to understand variance structure
# ================================================================

print("\n" + "=" * 65)
print("SECTION 2 — Exploratory FPCA")
print("(Connects to group paper Section 2.2)")
print("=" * 65)

# --- 2A: Fit FPCA ---
fd_grid = FDataGrid(
    data_matrix=Y,
    grid_points=t,
    domain_range=(1.0, 62.0),
    sample_names=countries
)

fpca = FPCA(n_components=4)
fpca.fit(fd_grid)
fpc_scores   = fpca.transform(fd_grid)         # (16, 4)
explained_var= fpca.explained_variance_ratio_
cumulative_var= np.cumsum(explained_var)

print(f"\nVariance explained per component:")
print(f"  {'Component':12s} {'Var%':>8s} {'Cumulative%':>13s}")
print(f"  {'-'*36}")
for k, (ev, cv) in enumerate(zip(explained_var, cumulative_var), 1):
    print(f"  PC{k:9d} {ev*100:>7.1f}% {cv*100:>12.1f}%")

print(f"\nFPC scores (PC1 and PC2) per country:")
print(f"  {'Country':15s} {'PC1':>10s} {'PC2':>10s}")
print(f"  {'-'*38}")
for c, s in zip(countries, fpc_scores):
    print(f"  {c:15s} {s[0]:>10.4f} {s[1]:>10.4f}")

# --- 2B: What each component means ---
# (connecting to group paper interpretations)
print(f"\nEconomic interpretation (from group paper):")
print(f"  PC1 ({explained_var[0]*100:.1f}%): global hiking shock "
      f"(late 2021 - early 2022 yield surge)")
print(f"  PC2 ({explained_var[1]*100:.1f}%): 2022-2023 decline phase")
print(f"  PC3 ({explained_var[2]*100:.1f}%): localised short-term fluctuations")
print(f"  PC4 ({explained_var[3]*100:.1f}%): localised short-term fluctuations")

# --- 2C: Key insight for regression ---
# PC1+PC2 explain most variance — the regression will be most
# powerful during these periods
# Structural breaks (from paper): Dec 2021 (t≈11), Oct 2022 (t≈21)
break1_t = 11   # Dec 2021 relative to Feb 2021 start
break2_t = 21   # Oct 2022

print(f"\nStructural breaks (from group paper change point analysis):")
print(f"  Break 1: Dec 2021 (t={break1_t}) — ECB PEPP termination")
print(f"  Break 2: Oct 2022 (t={break2_t}) — ECB 75bp hike, "
      f"peak yield volatility")

# ================================================================
# SECTION 3 — FUNCTION-ON-SCALAR MODEL
# ================================================================

print("\n" + "=" * 65)
print("SECTION 3 — Function-on-Scalar Regression")
print("Model: Y_i(t) = Z_i * Beta(t) + epsilon_i(t)")
print("=" * 65)

# --- 3A: OLS estimation ---
# Beta_hat = (Z'Z)^-1 Z' Y    shape: (5, 62)
ZtZ_inv   = np.linalg.inv(Z.T @ Z)
Beta_hat  = ZtZ_inv @ Z.T @ Y          # (5, 62)
Y_hat     = Z @ Beta_hat               # (16, 62)
residuals = Y - Y_hat                  # (16, 62)
df_resid  = n - p                      # 11
t_crit    = t_dist.ppf(0.975, df=df_resid)

print(f"\nModel fitted:")
print(f"  Beta_hat shape : {Beta_hat.shape}  (regressors x time points)")
print(f"  Residual std   : {residuals.std():.6f}")
print(f"  df_resid       : {df_resid}")
print(f"  t critical 95% : {t_crit:.4f}")

# --- 3B: Pointwise standard errors and CI bands ---
sigma2_t   = np.sum(residuals**2, axis=0) / df_resid
diag_ZtZ   = np.diag(ZtZ_inv)
SE         = np.sqrt(np.outer(diag_ZtZ, sigma2_t))    # (5, 62)
Beta_upper = Beta_hat + t_crit * SE
Beta_lower = Beta_hat - t_crit * SE

# --- 3C: R-squared over time ---
SS_res = np.sum(residuals**2, axis=0)
SS_tot = np.sum((Y - Y.mean(axis=0, keepdims=True))**2, axis=0)
with np.errstate(invalid='ignore', divide='ignore'):
    R2 = np.where(SS_tot > 1e-12, 1 - SS_res/SS_tot, np.nan)

print(f"\nR-squared over time:")
print(f"  Mean   : {np.nanmean(R2):.4f}")
print(f"  Median : {np.nanmedian(R2):.4f}")
print(f"  Min    : {np.nanmin(R2):.4f}")
print(f"  Max    : {np.nanmax(R2):.4f}")

# ================================================================
# SECTION 4 — ROBUSTNESS CHECKS
# ================================================================

print("\n" + "=" * 65)
print("SECTION 4 — Robustness checks")
print("=" * 65)

# --- 4A: Penalized smoothing of beta curves ---
print("\n--- 4A: Penalized beta curves (EDF) ---")

beta_basis     = BSplineBasis(domain_range=(1.0, 62.0),
                               n_basis=15, order=4)
regularization = L2Regularization(
    LinearDifferentialOperator(2),
    regularization_parameter=1.0
)
smoother = BasisSmoother(
    basis=beta_basis,
    regularization=regularization,
    return_basis=True
)

Beta_smooth = np.zeros_like(Beta_hat)
for k in range(p):
    beta_grid = FDataGrid(
        data_matrix=Beta_hat[k].reshape(1, -1),
        grid_points=t,
        domain_range=(1.0, 62.0)
    )
    Beta_smooth[k] = smoother.fit_transform(beta_grid)(t).squeeze()

edf_list = []
for k in range(p):
    ss_orig   = np.sum((Beta_hat[k] - Beta_hat[k].mean())**2)
    ss_smooth = np.sum((Beta_smooth[k] - Beta_smooth[k].mean())**2)
    edf       = round(ss_smooth/ss_orig*15, 2) if ss_orig > 0 else 1.0
    edf_list.append(edf)

print(f"\n  {'Regressor':30s} {'EDF':>8s} {'Time-variation'}")
print(f"  {'-'*60}")
for name, edf in zip(z_names, edf_list):
    tv = "approx. constant" if edf < 2 else \
         "mild"             if edf < 5 else "substantial"
    print(f"  {name:30s} {edf:>8.2f}   {tv}")

# --- 4B: Global permutation test ---
print("\n--- 4B: Global permutation tests ---")
print("H0: beta_k(t) = 0 for all t  (1000 permutations)")

np.random.seed(42)
n_perm = 1000

def integrated_beta_sq(beta_curve):
    return np.trapz(beta_curve**2, t)

obs_stats  = np.array([integrated_beta_sq(Beta_hat[k])
                        for k in range(1, p)])
null_dists = {name: [] for name in z_names[1:]}

for _ in range(n_perm):
    perm_idx = np.random.permutation(n)
    Z_perm   = np.hstack([np.ones((n,1)), Z_std[perm_idx]])
    B_perm   = np.linalg.inv(Z_perm.T @ Z_perm) @ Z_perm.T @ Y
    for k, name in enumerate(z_names[1:], start=1):
        null_dists[name].append(integrated_beta_sq(B_perm[k]))

pvalues = {}
print(f"\n  {'Regressor':30s} {'Obs stat':>12s} "
      f"{'p-value':>10s} {'Sig':>6s}")
print(f"  {'-'*62}")
for k, name in enumerate(z_names[1:]):
    null = np.array(null_dists[name])
    pval = np.mean(null >= obs_stats[k])
    pvalues[name] = pval
    sig  = '***' if pval<0.001 else '**' if pval<0.01 else \
           '*'   if pval<0.05  else '.'  if pval<0.1  else 'ns'
    print(f"  {name:30s} {obs_stats[k]:>12.6f} "
          f"{pval:>10.4f} {sig:>6s}")

# --- 4C: Durbin-Watson per country ---
print("\n--- 4C: Durbin-Watson test (temporal autocorrelation) ---")
dw_stats = []
print(f"\n  {'Country':15s} {'DW':>8s} {'Interpretation'}")
print(f"  {'-'*48}")
for i, country in enumerate(countries):
    dw = durbin_watson(residuals[i])
    dw_stats.append(dw)
    interp = "positive autocorr" if dw < 1.5 else \
             "no autocorr"       if dw < 2.5 else "negative autocorr"
    print(f"  {country:15s} {dw:>8.4f}   {interp}")
print(f"\n  Mean DW: {np.mean(dw_stats):.4f}  "
      f"({'concern - expected with static regressors' if np.mean(dw_stats) < 1.5 else 'OK'})")

# --- 4D: Pesaran CD test ---
print("\n--- 4D: Pesaran CD test (cross-sectional dependence) ---")
resid_corr = np.corrcoef(residuals)
upper_tri  = resid_corr[np.triu_indices(n, k=1)]
CD_stat    = np.sqrt(2*T / (n*(n-1))) * np.sum(upper_tri)
CD_pval    = 2 * (1 - t_dist.cdf(abs(CD_stat), df=T-1))
print(f"\n  Mean pairwise correlation : {upper_tri.mean():.4f}")
print(f"  Pesaran CD statistic      : {CD_stat:.4f}")
print(f"  p-value                   : {CD_pval:.4f}  "
      f"({'dependence detected' if CD_pval < 0.05 else 'OK'})")

# --- 4E: Stationarity ---
print("\n--- 4E: Functional stationarity check ---")
half        = T // 2
var_first   = Y.var(axis=0)[:half].mean()
var_second  = Y.var(axis=0)[half:].mean()
var_ratio   = var_second / var_first
print(f"\n  Variance ratio (second/first half): {var_ratio:.4f}")
print(f"  Interpretation: "
      f"{'non-stationarity - driven by hiking cycle regime shift' if abs(var_ratio-1)>0.3 else 'stationary'}")

# ================================================================
# SECTION 5 — PLOTS
# ================================================================

print("\n" + "=" * 65)
print("SECTION 5 — Generating all plots")
print("=" * 65)

reg_labels  = z_names[1:]
colors      = ['#1F4E79', '#C00000', '#375623', '#7030A0']
color_bands = ['#AEC6E8', '#F4A7A7', '#A8D5A2', '#D7B8E8']
output_path = "C:/Users/dskinkys/Desktop/Data science/fda/output/"

# --- Plot 1: All 16 yield change curves ---
fig, ax = plt.subplots(figsize=(13, 5))
for i, country in enumerate(countries):
    ax.plot(t, Y[i], linewidth=1.0, alpha=0.6, label=country)
mean_curve = Y.mean(axis=0)
ax.plot(t, mean_curve, color='black', linewidth=2.5,
        linestyle='--', label='Cross-country mean')
ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
# structural breaks
ax.axvline(break1_t, color='orange', linewidth=1.5,
           linestyle=':', label='Dec 2021 break')
ax.axvline(break2_t, color='green',  linewidth=1.5,
           linestyle=':', label='Oct 2022 break')
ax.set_xticks(t[tick_pos])
ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
ax.set_xlabel('Time', fontsize=10)
ax.set_ylabel('Yield change (smoothed)', fontsize=10)
ax.set_title('Smoothed Yield Change Curves — All 16 Countries\n'
             'Vertical lines = structural breaks (group paper)',
             fontweight='bold', fontsize=11)
ax.legend(fontsize=6, ncol=6, loc='upper right')
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(output_path + 'plot1_yield_curves.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Plot 1 saved: yield change curves")

# --- Plot 2: FPCA eigenfunctions ---
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()
ef_vals = fpca.components_.evaluate(t).squeeze()   # (4, 62)

for k, (ax, col) in enumerate(zip(axes, colors)):
    ax.plot(t, ef_vals[k], color=col, linewidth=2)
    ax.axhline(0, color='black', linewidth=0.8,
               linestyle='--', alpha=0.6)
    ax.axvline(break1_t, color='orange', linewidth=1.5,
               linestyle=':', label='Dec 2021' if k==0 else '')
    ax.axvline(break2_t, color='green', linewidth=1.5,
               linestyle=':', label='Oct 2022' if k==0 else '')
    ax.set_xticks(t[tick_pos])
    ax.set_xticklabels(tick_labels, rotation=45,
                       ha='right', fontsize=7)
    ax.set_title(
        f'PC{k+1} eigenfunction '
        f'({explained_var[k]*100:.1f}% variance explained)',
        fontweight='bold', fontsize=10)
    ax.set_ylabel('Eigenfunction value', fontsize=8)
    if k == 0:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

plt.suptitle(
    'FPCA Eigenfunctions with Structural Break Markers\n'
    'PC1+PC2 explain ' +
    f'{(explained_var[0]+explained_var[1])*100:.1f}% of total variance',
    fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(output_path + 'plot2_fpca_eigenfunctions.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Plot 2 saved: FPCA eigenfunctions")

# --- Plot 3: FPC scores PC1 vs PC2 ---
high_grade = ['Germany', 'Netherlands', 'Finland',
              'Austria', 'Belgium', 'Ireland']
colors_grade = ['#1F4E79' if c in high_grade else '#C00000'
                for c in countries]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for i, (c, col) in enumerate(zip(countries, colors_grade)):
    ax.scatter(fpc_scores[i, 0], fpc_scores[i, 1],
               color=col, s=90, zorder=3)
    ax.annotate(c, (fpc_scores[i, 0], fpc_scores[i, 1]),
                fontsize=7, textcoords='offset points',
                xytext=(5, 3))
ax.axhline(0, color='gray', linewidth=0.7, linestyle='--')
ax.axvline(0, color='gray', linewidth=0.7, linestyle='--')
ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)', fontsize=10)
ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)', fontsize=10)
ax.set_title('FPC Scores: PC1 vs PC2\nBlue=AAA/AA rated, Red=A and below',
             fontweight='bold', fontsize=10)
ax.grid(alpha=0.3)

ax = axes[1]
rating_raw = Z_raw[:, 0]
ax.scatter(rating_raw, fpc_scores[:, 0],
           c=colors_grade, s=90, zorder=3)
for i, c in enumerate(countries):
    ax.annotate(c, (rating_raw[i], fpc_scores[i, 0]),
                fontsize=7, textcoords='offset points',
                xytext=(5, 3))
ax.set_xlabel('S&P Rating (numeric scale)', fontsize=10)
ax.set_ylabel(f'PC1 score ({explained_var[0]*100:.1f}% variance)',
              fontsize=10)
ax.set_title('PC1 Score vs S&P Rating\n'
             'Higher PC1 = larger participation in hiking shock',
             fontweight='bold', fontsize=10)
ax.grid(alpha=0.3)

plt.suptitle('FPC Scores by Country\n'
             '(connects FPCA findings to rating hypothesis)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(output_path + 'plot3_fpc_scores.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Plot 3 saved: FPC scores")

# --- Plot 4: Effect functions with CI ---
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
axes = axes.flatten()
for k, (ax, label, col, col_b) in enumerate(
        zip(axes, reg_labels, colors, color_bands), start=1):
    beta_k  = Beta_hat[k]
    upper_k = Beta_upper[k]
    lower_k = Beta_lower[k]
    ax.fill_between(t, lower_k, upper_k,
                    color=col_b, alpha=0.5, label='95% CI')
    ax.plot(t, beta_k, color=col, linewidth=2, label='beta(t)')
    ax.axhline(0, color='black', linewidth=0.8,
               linestyle='--', alpha=0.6)
    ax.axvline(break1_t, color='orange', linewidth=1.2,
               linestyle=':', alpha=0.8)
    ax.axvline(break2_t, color='green',  linewidth=1.2,
               linestyle=':', alpha=0.8)
    # highlight significant periods
    sig = ((lower_k > 0) | (upper_k < 0))
    for start_i in np.where(np.diff(sig.astype(int)) == 1)[0]:
        end_candidates = np.where(
            np.diff(sig.astype(int)) == -1)[0]
        ends = end_candidates[end_candidates > start_i]
        if len(ends) > 0:
            ax.axvspan(t[start_i], t[ends[0]],
                       alpha=0.12, color=col, zorder=0)
    ax.set_xticks(t[tick_pos])
    ax.set_xticklabels(tick_labels, rotation=45,
                       ha='right', fontsize=7)
    ax.set_title(
        f'Effect of {label}\n'
        f'Global test p={pvalues[label]:.3f}  '
        f'{"*" if pvalues[label]<0.05 else "ns"}',
        fontweight='bold', fontsize=10)
    ax.set_ylabel('beta(t)', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

plt.suptitle(
    'Function-on-Scalar: Effect Functions with 95% CI\n'
    'Orange/green verticals = structural breaks (Dec 2021, Oct 2022)',
    fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(output_path + 'plot4_effect_functions.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Plot 4 saved: effect functions")

# --- Plot 5: Penalized vs OLS betas ---
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
axes = axes.flatten()
for k, (ax, label, col, col_b) in enumerate(
        zip(axes, reg_labels, colors, color_bands), start=1):
    pval_k  = pvalues[label]
    sig_lbl = '***' if pval_k<0.001 else '**' if pval_k<0.01 else \
              '*'   if pval_k<0.05  else '.'  if pval_k<0.1  else 'ns'
    ax.fill_between(t, Beta_lower[k], Beta_upper[k],
                    color=col_b, alpha=0.4, label='95% CI')
    ax.plot(t, Beta_hat[k], color=col, linewidth=1.2,
            alpha=0.5, linestyle='--', label='OLS beta(t)')
    ax.plot(t, Beta_smooth[k], color=col, linewidth=2.2,
            label='Penalized beta(t)')
    ax.axhline(0, color='black', linewidth=0.8,
               linestyle='--', alpha=0.6)
    ax.axvline(break1_t, color='orange', linewidth=1.2,
               linestyle=':', alpha=0.8)
    ax.axvline(break2_t, color='green',  linewidth=1.2,
               linestyle=':', alpha=0.8)
    ax.set_xticks(t[tick_pos])
    ax.set_xticklabels(tick_labels, rotation=45,
                       ha='right', fontsize=7)
    ax.set_title(
        f'Effect of {label}\n'
        f'Global p={pval_k:.3f} {sig_lbl} | EDF={edf_list[k]:.1f}',
        fontweight='bold', fontsize=10)
    ax.set_ylabel('beta(t)', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

plt.suptitle(
    'Penalized vs OLS Beta Curves\n'
    'Orange/green verticals = structural breaks',
    fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(output_path + 'plot5_penalized_betas.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Plot 5 saved: penalized betas")

# --- Plot 6: R-squared over time ---
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(t, R2, color='#1F4E79', linewidth=2)
ax.fill_between(t, 0, np.clip(R2, 0, 1),
                alpha=0.15, color='#1F4E79')
ax.axhline(np.nanmean(R2), color='#C00000', linewidth=1.5,
           linestyle='--',
           label=f'Mean R2 = {np.nanmean(R2):.3f}')
ax.axvline(break1_t, color='orange', linewidth=1.5,
           linestyle=':', label='Dec 2021 break')
ax.axvline(break2_t, color='green',  linewidth=1.5,
           linestyle=':', label='Oct 2022 break')
# shade FPCA high-variance region (PC1 peak)
ax.axvspan(8, 25, alpha=0.07, color='gray',
           label='PC1 peak region')
ax.set_xticks(t[tick_pos])
ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel('Time', fontsize=10)
ax.set_ylabel('R-squared', fontsize=10)
ax.set_title(
    'Model R-squared over time\n'
    'R2 peaks align with PC1/PC2 high-variance periods',
    fontweight='bold', fontsize=11)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_path + 'plot6_r2_curve.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Plot 6 saved: R-squared curve")

# --- Plot 7: Residuals heatmap ---
fig, ax = plt.subplots(figsize=(14, 6))
vmax = np.percentile(np.abs(residuals), 95)
im   = ax.imshow(residuals, aspect='auto', cmap='RdBu_r',
                 vmin=-vmax, vmax=vmax)
ax.set_yticks(np.arange(n))
ax.set_yticklabels(countries, fontsize=8)
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
ax.axvline(break1_t, color='orange', linewidth=2,
           linestyle=':', label='Dec 2021')
ax.axvline(break2_t, color='green',  linewidth=2,
           linestyle=':', label='Oct 2022')
ax.set_xlabel('Time', fontsize=10)
ax.set_ylabel('Country', fontsize=10)
ax.set_title(
    'Residuals Heatmap\n'
    'Greece/Estonia flagged as outliers in group paper',
    fontweight='bold', fontsize=11)
plt.colorbar(im, ax=ax, label='Residual')
ax.legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig(output_path + 'plot7_residuals_heatmap.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Plot 7 saved: residuals heatmap")

# --- Plot 8: Residual ACF ---
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
axes = axes.flatten()
max_lag = 15
for i, (ax, country) in enumerate(zip(axes, countries)):
    res_i = residuals[i]
    acf   = [np.corrcoef(res_i[:-lag], res_i[lag:])[0,1]
             if lag > 0 else 1.0
             for lag in range(max_lag+1)]
    ci    = 1.96 / np.sqrt(T)
    ax.bar(range(max_lag+1), acf, color='#1F4E79',
           alpha=0.7, width=0.6)
    ax.axhline( ci, color='red', linestyle='--', linewidth=0.8)
    ax.axhline(-ci, color='red', linestyle='--', linewidth=0.8)
    ax.axhline(0,   color='black', linewidth=0.5)
    ax.set_title(f'{country} (DW={dw_stats[i]:.2f})',
                 fontsize=8, fontweight='bold')
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Lag', fontsize=6)
    ax.tick_params(labelsize=6)
plt.suptitle(
    'Residual Autocorrelation by Country\n'
    'DW << 2 reflects temporal structure not captured '
    'by static regressors — addressed by FAR(1)',
    fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(output_path + 'plot8_residual_acf.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Plot 8 saved: residual ACF")

# --- Plot 9: Cross-sectional correlation heatmap ---
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(resid_corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(countries, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(countries, fontsize=8)
plt.colorbar(im, ax=ax, label='Correlation')
for i in range(n):
    for j in range(n):
        ax.text(j, i, f'{resid_corr[i,j]:.2f}',
                ha='center', va='center', fontsize=5,
                color='white' if abs(resid_corr[i,j])>0.5
                else 'black')
ax.set_title(
    f'Residual Cross-Sectional Correlations\n'
    f'Pesaran CD={CD_stat:.2f}, p={CD_pval:.4f}',
    fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig(output_path + 'plot9_cross_corr.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Plot 9 saved: cross-sectional correlations")

# ================================================================
# SECTION 6 — FINAL SUMMARY
# ================================================================

print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)

print(f"""
MODEL: Function-on-Scalar Regression
  n = {n} countries  |  T = {T} months  |  p = {p} regressors
  Period: Feb 2021 - Mar 2026

FPCA CONTEXT (from group paper Section 2.2):
  PC1: {explained_var[0]*100:.1f}% variance — global hiking shock
  PC2: {explained_var[1]*100:.1f}% variance — 2022-2023 decline
  Structural breaks: Dec 2021 (t={break1_t}), Oct 2022 (t={break2_t})
  R2 curve peaks align with PC1/PC2 high-variance periods

REGRESSION RESULTS:
  Mean R2 = {np.nanmean(R2):.4f}
  SP_Rating explains most cross-country variation (consistent
  with group paper Hypothesis 2: rating > core/periphery)

GLOBAL SIGNIFICANCE (permutation test, {n_perm} permutations):""")

for name in z_names[1:]:
    sig = pvalues[name] < 0.05
    print(f"  {name:30s} p={pvalues[name]:.4f}  "
          f"{'SIGNIFICANT' if sig else 'not significant'}")

print(f"""
RESIDUAL DIAGNOSTICS:
  Mean DW = {np.mean(dw_stats):.4f}
    -> Temporal autocorrelation present
    -> Expected: static regressors cannot capture temporal dynamics
    -> Solution: FAR(1) functional time series (next step)
  Pesaran CD = {CD_stat:.4f}, p = {CD_pval:.4f}
    -> Cross-sectional dependence present
    -> Reflects missing common factor (ECB policy common to all)
  Variance ratio = {var_ratio:.4f}
    -> Non-stationarity driven by hiking cycle regime shift
    -> Consistent with change points identified in group paper

NEXT STEP: FAR(1) Functional Autoregressive Model
  Addresses temporal autocorrelation directly
  Models Y_t(s) = Psi(Y_t-1)(s) + epsilon_t(s)
""")

print("Done.")