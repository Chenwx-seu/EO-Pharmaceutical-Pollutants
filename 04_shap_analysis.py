"""
05_shap_analysis.py
-------------------
SHAP TreeExplainer analysis for the best XGBoost model.
Produces a combined dot-plot + Mean(|SHAP|) bar chart.

Inputs  : best_model_XGBoost.pkl
          Dataset.xlsx 
Outputs : feature_importance_xgboost.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import joblib
import shap

# ── 0. Data ──────────────────────────────────────────────────────────────────
FILE_PATH      = 'Dataset.xlsx'
MODEL_PATH     = 'best_model_XGBoost.pkl'
SHAP_CLIP_LOW  = -5       # clip extreme negative SHAP values for plotting

df = pd.read_excel(FILE_PATH, sheet_name=2)

train_df = df[df['Original_Split'] == 'Train'].reset_index(drop=True)
test_df  = df[df['Original_Split'] == 'Test'].reset_index(drop=True)

X_train = train_df.iloc[:, 6:28] 
y_train = train_df['log2_K']
X_test  = test_df.iloc[:, 6:28]
y_test  = test_df['log2_K']


# ── 1. Load model & compute SHAP values ──────────────────────────────────────
best_model  = joblib.load(MODEL_PATH)
explainer   = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_train)

# ── 2. Rank features by mean |SHAP| ──────────────────────────────────────────
mean_shap    = np.abs(shap_values).mean(axis=0)
feature_names = X_train.columns

shap_df = pd.DataFrame({'feature': feature_names, 'mean_shap': mean_shap})\
            .sort_values('mean_shap', ascending=False)

sorted_features     = shap_df['feature'].values
orig_index          = [list(feature_names).index(f) for f in sorted_features]
shap_values_sorted  = shap_values[:, orig_index]
mean_shap_sorted    = shap_df['mean_shap'].values
mean_shap_percent   = mean_shap_sorted / mean_shap_sorted.sum() * 100

X_sorted = X_train[sorted_features]

# ── 3. Build figure ───────────────────────────────────────────────────────────
shap_filtered = shap_values_sorted.copy()
shap_filtered[shap_filtered < SHAP_CLIP_LOW] = np.nan

shap.summary_plot(shap_filtered, X_sorted, plot_type="dot", show=False)

fig = plt.gcf()
fig.set_size_inches(12, 6)
fig.subplots_adjust(left=0.2, right=0.8)

ax_sw = plt.gca()
ax_sw.set_xlim(-4, 4)
ax_sw.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# Overlay bar chart on a twin x-axis (behind the dot plot)
ax_bar = ax_sw.twiny()
ax_bar.set_zorder(0)
ax_sw.set_zorder(1)
ax_sw.patch.set_alpha(0)

n_feat  = len(sorted_features)
y_pos   = np.arange(n_feat)[::-1]
xlim_bar = mean_shap_sorted.max() * 1.05

ax_bar.barh(y=y_pos, width=mean_shap_sorted, height=0.6,
            color='lightpink', alpha=0.5, edgecolor='none')
ax_bar.set_xlim(0, xlim_bar)

xticks_bar = np.linspace(0, xlim_bar, 5)
ax_bar.set_xticks(xticks_bar)
ax_bar.set_xticklabels([f"{x:.2f}" for x in xticks_bar])
ax_bar.set_xlabel("Mean(|SHAP|)", fontsize=10, fontweight='bold')
ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(sorted_features)

for i, (v, p) in enumerate(zip(mean_shap_sorted, mean_shap_percent)):
    ax_bar.text(0.01 * xlim_bar, y_pos[i],
                f"{v:.3f} ({p:.1f}%)",
                va='center', ha='left', fontsize=10, color='black')

plt.tight_layout()
plt.savefig("feature_importance_xgboost.png", dpi=400,
            bbox_inches='tight', format='png')
plt.show()
print("Saved: feature_importance_xgboost.png")
