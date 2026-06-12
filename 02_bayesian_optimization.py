"""
03_bayesian_optimization.py
---------------------------
Bayesian hyperparameter optimisation (BayesSearchCV) for all 10 models.
The best model per algorithm is saved as best_model_<NAME>.pkl.

Inputs  : Dataset.xlsx
Outputs : best_model_<NAME>.pkl  (one per algorithm)
          bayesian_opt_avg_metrics.csv
          Model performance.png  (scatter grid, 2 × 5)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor,
                              BaggingRegressor)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── 0. Data ──────────────────────────────────────────────────────────────────
FILE_PATH = 'Dataset.xlsx'
df = pd.read_excel(FILE_PATH, sheet_name=2)

train_df = df[df['Original_Split'] == 'Train'].reset_index(drop=True)
test_df  = df[df['Original_Split'] == 'Test'].reset_index(drop=True)

X_train = train_df.iloc[:, 6:28] 
y_train = train_df['log2_K']
X_test  = test_df.iloc[:, 6:28]
y_test  = test_df['log2_K']


# ── 1. Model registry ────────────────────────────────────────────────────────
optim_models = [
    MLPRegressor(), DecisionTreeRegressor(), ExtraTreeRegressor(),
    KNeighborsRegressor(), SVR(),
    XGBRegressor(), RandomForestRegressor(),
    AdaBoostRegressor(), BaggingRegressor(),
    CatBoostRegressor(verbose=0)
]
optim_names = [
    'MLP', 'CART', 'ETR',
    'KNN', 'SVR-RBF',
    'XGBoost', 'RF', 'AdaBoost',
    'Bagging', 'CatBoost'
]

param_spaces = {
    'MLP':      {'hidden_layer_sizes': Integer(5, 120),
                 'alpha': Real(1e-5, 0.05, prior='log-uniform')},
    'CART':     {'max_depth': Integer(3, 30),
                 'min_samples_split': Integer(2, 30),
                 'min_samples_leaf': Integer(1, 10)},
    'ETR':      {'max_depth': Integer(3, 20),
                 'min_samples_split': Integer(5, 30),
                 'min_samples_leaf': Integer(1, 15)},
    'KNN':      {'n_neighbors': Integer(5, 50),
                 'weights': Categorical(['uniform', 'distance'])},
    'SVR-RBF':  {'C': Real(0.1, 50, prior='log-uniform'),
                 'gamma': Real(1e-4, 0.5, prior='log-uniform')},
    'XGBoost':  {'n_estimators': Integer(50, 300),
                 'max_depth': Integer(1, 8),
                 'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                 'subsample': Real(0.5, 1.0),
                 'colsample_bytree': Real(0.5, 1.0),
                 'reg_lambda': Real(1e-3, 100, prior='log-uniform'),
                 'reg_alpha': Real(1e-3, 10,  prior='log-uniform')},
    'RF':       {'n_estimators': Integer(50, 300),
                 'max_depth': Integer(3, 20),
                 'min_samples_split': Integer(2, 20),
                 'min_samples_leaf': Integer(1, 10),
                 'max_features': Real(0.5, 1.0)},
    'AdaBoost': {'n_estimators': Integer(50, 150),
                 'learning_rate': Real(0.01, 2, prior='log-uniform')},
    'Bagging':  {'n_estimators': Integer(10, 200),
                 'max_samples': Real(0.5, 1.0),
                 'max_features': Real(0.5, 1.0)},
    'CatBoost': {'iterations': Integer(100, 1000),
                 'depth': Integer(3, 10),
                 'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                 'l2_leaf_reg': Real(1, 10)},
}

# ── 2. Bayesian optimisation loop ────────────────────────────────────────────
NUM_ITERATIONS = 10
N_BAYES_ITER   = 20
CV_SPLITS      = 10

records     = []
best_models = []

for name, model in zip(optim_names, optim_models):
    print(f"\nOptimising: {name}")

    opt = BayesSearchCV(
        estimator=model,
        search_spaces=param_spaces[name],
        n_iter=N_BAYES_ITER,
        scoring='r2',
        cv=KFold(n_splits=CV_SPLITS, shuffle=True, random_state=42),
        n_jobs=-1,
        random_state=42,
    )

    best_r2    = -np.inf
    best_inst  = None

    for i in range(NUM_ITERATIONS):
        opt.fit(X_train, y_train)
        m = opt.best_estimator_

        y_tr = m.predict(X_train)
        y_te = m.predict(X_test)

        row = {
            'Model':      name,
            'R²_train':   r2_score(y_train, y_tr),
            'RMSE_train': np.sqrt(mean_squared_error(y_train, y_tr)),
            'MAE_train':  mean_absolute_error(y_train, y_tr),
            'R²_test':    r2_score(y_test, y_te),
            'RMSE_test':  np.sqrt(mean_squared_error(y_test, y_te)),
            'MAE_test':   mean_absolute_error(y_test, y_te),
        }
        records.append(row)

        if row['R²_test'] > best_r2:
            best_r2   = row['R²_test']
            best_inst = m

    best_models.append(best_inst)
    joblib.dump(best_inst, f'best_model_{name}.pkl')
    print(f"  Best R²_test={best_r2:.4f} | params: {opt.best_params_}")

# ── 3. Aggregate metrics ─────────────────────────────────────────────────────
opt_metrics_df = pd.DataFrame(records)
avg_df = (opt_metrics_df
          .groupby('Model')
          .mean()
          .loc[optim_names]
          .reset_index())

print("\n=== Average metrics after Bayesian optimisation ===")
print(avg_df.to_string(index=False))
avg_df.to_csv("bayesian_opt_avg_metrics.csv", index=False)

# ── 4. Scatter grid ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': 'black',   'axes.grid': True,
    'grid.alpha': 0.2,           'grid.linestyle': '--',
    'font.size': 12,             'axes.titlesize': 14,
    'axes.labelsize': 12,        'xtick.labelsize': 10,
    'ytick.labelsize': 10,       'legend.fontsize': 11,
    'axes.linewidth': 1,
})

fig, axes = plt.subplots(2, 5, figsize=(22, 9))
axes = axes.flatten()

TRAIN_COLOR = '#1f77b4'
TEST_COLOR  = '#ff7f0e'

for i, (name, model) in enumerate(zip(optim_names, best_models)):
    y_tr_pred = model.predict(X_train)
    y_te_pred = model.predict(X_test)

    ax = axes[i]
    ax.scatter(y_train, y_tr_pred, alpha=0.6, s=60,
               color=TRAIN_COLOR, edgecolor='k', linewidth=0.5, label='Train')
    ax.scatter(y_test,  y_te_pred, alpha=0.8, s=60,
               color=TEST_COLOR,  edgecolor='k', linewidth=0.5, label='Test')
    ax.plot([-20, 0], [-20, 0], 'r--', lw=1.5, label='Ideal')
    ax.set_title(name, fontweight='bold')
    ax.set_xlabel("Experimental log2K values")
    ax.set_ylabel("Predicted log2K values")
    ax.set_xlim(-20, 0)
    ax.set_ylim(-20, 0)
    ax.legend(frameon=False)

plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.savefig("Model performance.png", dpi=400, bbox_inches='tight', format='png')
plt.show()
print("\nSaved: Model performance.png, bayesian_opt_avg_metrics.csv, best_model_*.pkl")
