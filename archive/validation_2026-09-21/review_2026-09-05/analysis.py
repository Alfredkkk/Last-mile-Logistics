
# CELL 1
# =========================================
# Common Initialization: Imports, Configurations, and Helper Functions
# =========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Global Parameters ----------
CSV_PATH = "Results/param_sweep_results_2.csv" #For the full parameter test with lambda=40 and all alpha, use param_sweep_results_2_full_param.csv

# Color definitions
COLOR_SWITCH = "#1a80bb"   # HVOR (Switching) blue
COLOR_PURE   = "#ea801c"   # PURE_OR orange
COLOR_SOLID  = "#1a80bb"   # Solid line color
COLOR_DASHED = "#000000"   # Dashed line color

# Line style definitions
DASH_DOT = (0, (6, 2, 1.5, 2))  # dash-dot style
DASHED = (0, (6, 2))            # dashed style
# ----------------------------------------

# Paper-style matplotlib configuration
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

# Common helper function
def paper_axes(ax):
    """Make axes look paper-style: four borders, inward ticks, no grid."""
    ax.set_frame_on(True)
    for side in ("left", "right", "top", "bottom"):
        spine = ax.spines[side]
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor("black")
        spine.set_position(("outward", 0))
    ax.tick_params(which="both", direction="in", width=1.0, length=4,
                   top=True, right=True)
    ax.grid(False)

def read_training_log(path):
    import csv
    rows = []
    with open(path, newline='') as f:
        r = csv.reader(f)
        header = next(r, [])
        if not header:
            return pd.DataFrame()
        for row in r:
            if len(row) > len(header):
                header.extend([f'extra_{i}' for i in range(len(header)+1, len(row)+1)])
            if len(row) < len(header):
                row = row + [''] * (len(header) - len(row))
            rows.append(row)
    df = pd.DataFrame(rows, columns=header)
    num_cols = [
        'update','reward','rate','ep_rate','terminal_time_min','steps','episodes',
        'step_avg_reward','step_avg_rate','LAMBDA','GAMMA_PACK','R_PICK_ALPHA',
        'RIDE_TTL_MIN','MAX_VISIBLE_RIDES','SWITCH_GRACE_STEPS','RT','combo_id'
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'kind' not in df.columns:
        df['kind'] = 'train'
    return df


# CELL 2

# =========================================
# Paper-style: Revenue rate vs λ (two separate plots)
# =========================================

def best_alpha_series(df_in: pd.DataFrame, gamma_val, algo_name):
    """For each λ, select R_PICK_ALPHA that gives the highest rate (consistent with paper's optimal α)"""
    sub = df_in[(df_in["GAMMA_PACK"] == gamma_val) & (df_in["algo"] == algo_name)]
    if sub.empty:
        return np.array([]), np.array([])
    grp = (sub.groupby(["LAMBDA", "R_PICK_ALPHA"])["rate"]
              .mean()
              .reset_index())
    best = grp.loc[grp.groupby("LAMBDA")["rate"].idxmax()].sort_values("LAMBDA")
    return best["LAMBDA"].to_numpy(), best["rate"].to_numpy()

# Load and filter data
df = pd.read_csv(CSV_PATH)
df = df[df["RT"] == 6.0].copy()
df = df[df["algo"].isin(["HEUR_VOR", "PURE_OR"])].copy()

# Two separate plots: n=30→γ≈0.50, n=50→γ≈0.83
PANELS = [
    {"title": r"$n=30,\ r_\ell=6$", "gamma": 0.50, "filename": "rev_rate_vs_lambda_n30.png"},
    {"title": r"$n=50,\ r_\ell=6$", "gamma": 0.83, "filename": "rev_rate_vs_lambda_n50.png"},
]

for p in PANELS:
    g = p["gamma"]
    
    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(5.9, 4.6), dpi=160)

    # Get data (optimal α for each λ)
    lam_h, rate_h = best_alpha_series(df, g, "HEUR_VOR")
    lam_p, rate_p = best_alpha_series(df, g, "PURE_OR")

    # Plot lines
    if lam_h.size:
        ax.plot(lam_h, rate_h, color=COLOR_SWITCH, lw=2.0, label="Switching")
    if lam_p.size:
        ax.plot(lam_p, rate_p, color=COLOR_PURE, lw=2.0, linestyle=DASH_DOT,
                label="Pure delivery")

    # Axis styling
    paper_axes(ax)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"Revenue rate ($\$/minute$)")
    ax.set_title(p["title"])

    # Legend: inside right, no frame, longer lines
    ax.legend(loc="center right", frameon=False, handlelength=3.2,
              borderaxespad=1.2)

    # Set reasonable y margin to avoid top edge
    y_candidates = []
    if rate_h.size: y_candidates += [rate_h.min(), rate_h.max()]
    if rate_p.size: y_candidates += [rate_p.min(), rate_p.max()]
    if y_candidates:
        ylo, yhi = min(y_candidates), max(y_candidates)
        pad = max(0.01, 0.04 * (yhi - ylo))
        ax.set_ylim(ylo - pad, yhi + pad)

    fig.tight_layout()
    plt.show()
    
    # Save figure
    fig.savefig(p["filename"], dpi=300, bbox_inches="tight")
    print(f"Saved: {p['filename']}")

# CELL 3
# =========================================
# Paper-style: Revenue rate vs n (two separate plots for λ)
# =========================================

RT_FILTER = 6.0
R_ELL = f"{RT_FILTER:g}"  # r_ℓ for display
LAMBDAS = [5, 10, 20, 30, 40]  # Lambda values for panels
RATE_COL = "rate"  # Main metric: avg_rate = sum(reward) / sum(terminal_time); ep_rate is diagnostic only.

def best_alpha_by_lambda_gamma(df_in: pd.DataFrame, lam, gamma, algo):
    """Fix λ, γ, algo, and select α that gives the maximum time-weighted avg_rate"""
    sub = df_in[(df_in["LAMBDA"] == lam) &
                (df_in["GAMMA_PACK"] == gamma) &
                (df_in["algo"] == algo)]
    if sub.empty:
        return np.nan
    # Average over same α first, then take maximum
    alpha_mean = sub.groupby("R_PICK_ALPHA")[RATE_COL].mean()
    return float(alpha_mean.max())

# Load data, keep only the two required curves
df = pd.read_csv(CSV_PATH)
df = df[df["RT"] == RT_FILTER].copy()
df = df[df["algo"].isin(["HEUR_VOR", "PURE_OR"])].copy()

# γ to n conversion (n ≈ 60*γ)
GAMMAS = [0.33, 0.50, 0.67, 0.83, 1.00]
N_LIST = [int(round(g*60)) for g in GAMMAS]  # -> [20,30,40,50,60]

for lam in LAMBDAS:
    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(5.9, 4.6), dpi=160)
    
    # Get optimal α rate for each n (corresponding γ)
    rate_hvor = []
    rate_puro = []
    for g in GAMMAS:
        rate_hvor.append(best_alpha_by_lambda_gamma(df, lam, g, "HEUR_VOR"))
        rate_puro.append(best_alpha_by_lambda_gamma(df, lam, g, "PURE_OR"))

    # Plot lines
    ax.plot(N_LIST, rate_hvor, color=COLOR_SWITCH, lw=2.0, label="Switching")
    ax.plot(N_LIST, rate_puro, color=COLOR_PURE, lw=2.0, linestyle=DASH_DOT, label="Pure delivery")

    # Axis styling and labels
    paper_axes(ax)
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"Revenue rate ($\$/minute$)")
    ax.set_title(rf"$\lambda={lam},\ r_\ell={R_ELL}$")

    # Legend: inside lower right, no frame
    ax.legend(loc="lower right", frameon=False, handlelength=3.2)

    # Set appropriate padding
    y_all = [v for v in rate_hvor + rate_puro if np.isfinite(v)]
    if y_all:
        ylo, yhi = min(y_all), max(y_all)
        pad = max(0.01, 0.05*(yhi - ylo if yhi > ylo else 0.2))
        ax.set_ylim(ylo - pad, yhi + pad)

    fig.tight_layout()
    plt.show()
    
    # Save figure
    filename = f"rev_rate_vs_n_lambda{lam}.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved: {filename}")

# CELL 4
# =========================================
# Paper-style: Ratio HVOR/DRL vs λ and vs n (two separate plots)
# =========================================

RT_FILTER = 6.0  # Only use results with r_ℓ = 6
LAM_LOW, LAM_HIGH = 10, 40  # Low/high λ for right plot
RATE_COL = "rate"  # Main metric: avg_rate = sum(reward) / sum(terminal_time); ep_rate is diagnostic only.

def best_alpha_rate(df_in, lam, gamma, algo):
    """Fix λ, γ, algo, and select the best α by maximum time-weighted avg_rate."""
    sub = df_in[(df_in["LAMBDA"] == lam) &
                (df_in["GAMMA_PACK"] == gamma) &
                (df_in["algo"] == algo)]
    if sub.empty:
        return np.nan
    alpha_mean = sub.groupby("R_PICK_ALPHA")[RATE_COL].mean()
    return float(alpha_mean.max())

# Load and filter by r_ℓ, keep only DRL and HEUR_VOR
df = pd.read_csv(CSV_PATH)
df = df[df["RT"] == RT_FILTER]
df = df[df["algo"].isin(["DRL", "HEUR_VOR"])].copy()

# γ to n conversion (n ≈ 60*γ)
GAMMAS_ALL = sorted(df["GAMMA_PACK"].unique())
N_ALL = [int(round(g*60)) for g in GAMMAS_ALL]
gamma_of_n = {int(round(g*60)): g for g in GAMMAS_ALL}

# ---------- Plot 1: ratio vs λ (two curves: n=30, n=50) ----------
fig, ax = plt.subplots(1, 1, figsize=(5.9, 4.6), dpi=160)

lambda_vals = sorted(df["LAMBDA"].unique())
n_left_curves = [(30, COLOR_SOLID, "solid", "n = 30"),
                 (50, COLOR_DASHED, DASHED, "n = 50")]

for n_val, col, ls, lab in n_left_curves:
    g = gamma_of_n.get(n_val, None)
    if g is None:
        continue
    xs, ys = [], []
    for lam in lambda_vals:
        if lam < 2:  # Skip λ values less than 2
            continue
        hv = best_alpha_rate(df, lam, g, "HEUR_VOR")
        dr = best_alpha_rate(df, lam, g, "DRL")
        if np.isfinite(hv) and np.isfinite(dr) and dr > 0:
            xs.append(lam); ys.append(hv/dr)
    ax.plot(xs, ys, color=col, lw=2.0, linestyle=ls, label=lab)

paper_axes(ax)
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$R_{\mathrm{Switching}}/R_{\mathrm{DRL}}$")
ax.legend(loc="lower right", frameon=False)

fig.tight_layout()
plt.show()

# Save figure
filename = "ratio_hvor_drl_vs_lambda.png"
fig.savefig(filename, dpi=300, bbox_inches="tight")
print(f"Saved: {filename}")

# ---------- Plot 2: ratio vs n (two curves: λ=40, λ=5) ----------
fig, ax = plt.subplots(1, 1, figsize=(5.9, 4.6), dpi=160)

lam_right_curves = [(LAM_HIGH, COLOR_SOLID, "solid", rf"$\lambda={LAM_HIGH}$"),
                    (LAM_LOW,  COLOR_DASHED, DASHED, rf"$\lambda={LAM_LOW}$")]

for lam, col, ls, lab in lam_right_curves:
    xs, ys = [], []
    for n_val in sorted(gamma_of_n.keys()):
        g = gamma_of_n[n_val]
        hv = best_alpha_rate(df, lam, g, "HEUR_VOR")
        dr = best_alpha_rate(df, lam, g, "DRL")
        if np.isfinite(hv) and np.isfinite(dr) and dr > 0:
            xs.append(n_val); ys.append(hv/dr)
    ax.plot(xs, ys, color=col, lw=2.0, linestyle=ls, label=lab)

paper_axes(ax)
ax.set_xlabel(r"$n$")
ax.set_ylabel(r"$R_{\mathrm{Switching}}/R_{\mathrm{DRL}}$")
ax.legend(loc="lower right", frameon=False)

fig.tight_layout()
plt.show()

# Save figure
filename = "ratio_hvor_drl_vs_n.png"
fig.savefig(filename, dpi=300, bbox_inches="tight")
print(f"Saved: {filename}")

# CELL 5
# =========================================
# DRL convergence using current training_log.csv (train vs eval)
import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

log_path = Path('Results/training_log.csv')
if not log_path.exists():
    raise FileNotFoundError('Results/training_log.csv not found. Run experiment.ipynb after the fixed logger changes to generate a fresh log.')

df = pd.read_csv(log_path)
if 'kind' not in df.columns:
    raise ValueError('training_log.csv lacks the kind column; regenerate it with the fixed logger instead of using legacy logs.')


df['update'] = df['update'].astype(int)
train_df = df[df['kind'] == 'train']
eval_df = df[df['kind'] == 'eval']

# Training: use step_avg_rate, which includes incomplete rollout fragments, then average and smooth by update.
if not train_df.empty:
    agg_train = train_df.groupby('update')['step_avg_rate'].mean().reset_index()
    agg_train['smooth'] = agg_train['step_avg_rate'].rolling(window=3, min_periods=1).mean()
    plt.figure(figsize=(6, 4))
    plt.plot(agg_train['update'], agg_train['step_avg_rate'], marker='o', alpha=0.4, label='Train (mean)')
    plt.plot(agg_train['update'], agg_train['smooth'], linewidth=2, label='Train (rolling mean)')
    plt.xlabel('PPO update')
    plt.ylabel('Revenue rate (scaled)')
    plt.title('DRL training convergence (step_avg_rate)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

# Evaluation: use eval-row avg_rate, then average and smooth by update.
if not eval_df.empty:
    agg_eval = eval_df.groupby('update')['rate'].mean().reset_index()
    agg_eval['smooth'] = agg_eval['rate'].rolling(window=3, min_periods=1).mean()
    plt.figure(figsize=(6, 4))
    plt.plot(agg_eval['update'], agg_eval['rate'], marker='o', alpha=0.4, label='Eval (mean)')
    plt.plot(agg_eval['update'], agg_eval['smooth'], linewidth=2, label='Eval (rolling mean)')
    plt.xlabel('PPO update')
    plt.ylabel('Revenue rate (scaled)')
    plt.title('DRL evaluation convergence (evaluate_all)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


# CELL 6
# The comparison of the average terminal time between algorithms

# Load the dataset
csv_path = 'Results/param_sweep_results_2.csv'
df = pd.read_csv(csv_path)

# Filter for HEUR and HEUR_VOR algorithms
heur_df = df[df['algo'].isin(['DRL', 'HEUR', 'HEUR_VOR', 'PURE', 'PURE_OR'])]

# Calculate and print the mean terminal_time for each
mean_terminal_time = heur_df.groupby('algo')['terminal_time'].mean()

print("Average Terminal Time Comparison:")
print(mean_terminal_time)



# CELL 7

# =========================================
# Avg terminal time by n/lambda (grouped bar charts)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path = 'Results/param_sweep_results_2.csv'
df = pd.read_csv(csv_path)
R_val = 5.5
df['n_est'] = df['GAMMA_PACK'] * (2 * (R_val ** 2))
# round n to nearest {20,30,40,50,60}
targets = np.array([20, 30, 40, 50, 60], dtype=float)
df['n_round'] = df['n_est'].apply(lambda x: targets[np.argmin(np.abs(targets - x))])

algos = ['DRL','HEUR','HEUR_VOR','FOUR_ZONE','PURE','PURE_OR']
df = df[df['algo'].isin(algos)].copy()

# --- By lambda: grouped bars over n ---
lambdas = sorted(df['LAMBDA'].unique())
for lam in lambdas:
    sub = df[df['LAMBDA'] == lam]
    grp = (sub.groupby(['algo','n_round'])['terminal_time']
             .mean()
             .reset_index())
    n_vals = sorted(grp['n_round'].unique())
    x = np.arange(len(n_vals))
    width = 0.12
    plt.figure(figsize=(7,4))
    for i, algo in enumerate(algos):
        sg = grp[grp['algo']==algo]
        y = [sg[sg['n_round']==n]['terminal_time'].values[0] if n in sg['n_round'].values else np.nan for n in n_vals]
        plt.bar(x + (i - len(algos)/2)*width + width/2, y, width=width, label=algo)
    plt.title(f'Lambda = {lam}')
    plt.xlabel('n (expected package count)')
    plt.ylabel('Avg terminal time (min)')
    plt.xticks(x, [f'{n:.1f}' for n in n_vals])
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.show()

# --- By n: grouped bars over lambda ---
n_levels = sorted(df['n_round'].unique())
for n in n_levels:
    sub = df[df['n_round'] == n]
    grp = (sub.groupby(['algo','LAMBDA'])['terminal_time']
             .mean()
             .reset_index())
    lam_vals = sorted(grp['LAMBDA'].unique())
    x = np.arange(len(lam_vals))
    width = 0.12
    plt.figure(figsize=(7,4))
    for i, algo in enumerate(algos):
        sg = grp[grp['algo']==algo]
        y = [sg[sg['LAMBDA']==lam]['terminal_time'].values[0] if lam in sg['LAMBDA'].values else np.nan for lam in lam_vals]
        plt.bar(x + (i - len(algos)/2)*width + width/2, y, width=width, label=algo)
    plt.title(f'n = {n:.1f}')
    plt.xlabel('Lambda')
    plt.ylabel('Avg terminal time (min)')
    plt.xticks(x, [str(l) for l in lam_vals])
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.show()


# CELL 8
# =========================================
# Scatter: termination time vs revenue rate (n=30/50, lambda=10/40)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_path = 'Results/param_sweep_results_2.csv'
df = pd.read_csv(csv_path)
R_val = 5.5
df['n_est'] = df['GAMMA_PACK'] * (2 * (R_val ** 2))
targets = np.array([30, 50], dtype=float)
df['n_round'] = df['n_est'].apply(lambda x: targets[np.argmin(np.abs(targets - x))])

policies = {'PURE':'Pure delivery', 'HEUR_VOR':'Switching', 'DRL':'DRL'}
lambdas = [10, 40]
n_levels = [30, 50]
RATE_COL = "rate"  # Main metric: avg_rate = sum(reward) / sum(terminal_time); ep_rate is diagnostic only.

fig, axes = plt.subplots(len(n_levels), len(lambdas), figsize=(10, 6), sharex=False, sharey=False)
for i, n in enumerate(n_levels):
    for j, lam in enumerate(lambdas):
        ax = axes[i][j] if len(n_levels) > 1 else axes[j]
        sub = df[(df['n_round'] == n) & (df['LAMBDA'] == lam) & (df['algo'].isin(policies.keys()))]
        for algo, label in policies.items():
            sg = sub[sub['algo'] == algo]
            ax.scatter(sg['terminal_time'], sg[RATE_COL], label=label, alpha=0.7)
        ax.set_title(f'n={n}, lambda={lam}')
        ax.set_xlabel('Terminal time (min)')
        ax.set_ylabel('Revenue rate (avg_rate)')
        ax.grid(True, alpha=0.3)
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
plt.tight_layout()
plt.show()
