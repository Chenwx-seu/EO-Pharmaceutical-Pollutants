"""
01_feature_selection.py
-----------------------
Two-stage feature selection pipeline:
  Stage 1 – Mutual Information (MI) filtering  (threshold = 0.01)
  Stage 2 – RFECV with RandomForest + tolerance-based optimal point

Inputs  : Dataset.xlsx
Outputs : optimal_features.csv
          RFECV_Green_Point_Features.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression, RFECV, RFE
from sklearn.model_selection import KFold

# ── 0. Data ──────────────────────────────────────────────────────────────────
FILE_PATH = 'Dataset.xlsx'
df = pd.read_excel(FILE_PATH, sheet_name=1)

train_df = df[df['Original_Split'] == 'Train'].reset_index(drop=True)
test_df  = df[df['Original_Split'] == 'Test'].reset_index(drop=True)

X_train = train_df.iloc[:, 6:161] 
y_train = train_df['log2_K']
X_test  = test_df.iloc[:, 6:161]
y_test  = test_df['log2_K']

# ── 2. Stage 1 – Mutual Information filtering ────────────────────────────────
mi_scores  = mutual_info_regression(X_train, y_train, random_state=42)
mi_series  = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)
MI_THRESHOLD = 0.01
selected_mi = mi_series[mi_series > MI_THRESHOLD].index.tolist()
X_train_mi = X_train[selected_mi]
# ── 3. Stage 2 – RFECV ───────────────────────────────────────────────────────
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
cv = KFold(n_splits=10, shuffle=True, random_state=42)
rfecv = RFECV(estimator=rf_model, step=1, cv=cv, scoring='r2', n_jobs=-1)
rfecv.fit(X_train_mi, y_train)
scores      = rfecv.cv_results_['mean_test_score']
num_features = np.arange(1, len(scores) + 1)
# Tolerance-based optimal: minimum features within 2 % of max R²
max_score          = scores.max()
TOLERANCE          = 0.02
within_tol_idx     = np.where(scores >= max_score * (1 - TOLERANCE))[0]
optimal_idx        = within_tol_idx[0]          # fewest features inside tolerance
optimal_n_features = optimal_idx + 1

# ── 4. Plot RFECV curve ───────────────────────────────────────────────────────
plt.figure(figsize=(7, 6))
plt.plot(num_features, scores, color='blue', linestyle='-', alpha=0.8)
plt.scatter(num_features, scores, color='red', alpha=0.8)
plt.scatter(optimal_n_features, scores[optimal_idx], color='green', s=100,
            label="Optimal features")
plt.axvline(x=optimal_n_features, color='green', linestyle='--', linewidth=1.2)
plt.axvline(x=22, color='black', linestyle='--', linewidth=1.2)
plt.grid(False)
plt.xlabel("Number of features",
           fontname='Times New Roman', fontsize=14, fontweight='bold')
plt.ylabel("R$^2$ value of CV",
           fontname='Times New Roman', fontsize=14, fontweight='bold')
plt.xticks(fontname='Times New Roman', fontsize=14, fontweight='bold')
plt.yticks(fontname='Times New Roman', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', frameon=False,
           prop=font_manager.FontProperties(
               family='Times New Roman', size=14, weight='bold'))
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.2)
plt.tight_layout()
plt.show()

# ── 5. Extract selected features with RFE ───────────────────────────────────
rfe_manual = RFE(estimator=rf_model, n_features_to_select=optimal_n_features, step=1)
rfe_manual.fit(X_train_mi, y_train)
selected_features = X_train_mi.columns[rfe_manual.support_]
pd.DataFrame({'feature': list(selected_features)}).to_csv(
    "optimal_features.csv", index=False)
selected_features.to_series().to_csv(
    "RFECV_Green_Point_Features.csv", index=False)
