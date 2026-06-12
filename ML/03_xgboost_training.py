"""
04_xgboost_training.py
----------------------
Load the Bayesian-optimised XGBoost model, re-fit on train, evaluate on both
sets, and export predictions to Excel.

Inputs  : best_model_XGBoost.pkl
          Dataset.xlsx
Outputs : XGBoost_Train_Predictions.xlsx
          XGBoost_Test_Predictions.xlsx
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── 0. Data ──────────────────────────────────────────────────────────────────
FILE_PATH = 'Dataset.xlsx'
df = pd.read_excel(FILE_PATH, sheet_name=2)

train_df = df[df['Original_Split'] == 'Train'].reset_index(drop=True)
test_df  = df[df['Original_Split'] == 'Test'].reset_index(drop=True)

X_train = train_df.iloc[:, 6:28] 
y_train = train_df['log2_K']
X_test  = test_df.iloc[:, 6:28]
y_test  = test_df['log2_K']


# ── 1. Load & fit model ───────────────────────────────────────────────────────
model = joblib.load("best_model_XGBoost.pkl")
model.fit(X_train, y_train)

# ── 2. Predict ────────────────────────────────────────────────────────────────
y_train_pred = model.predict(X_train)
y_test_pred  = model.predict(X_test)

# ── 3. Metrics ────────────────────────────────────────────────────────────────
for split, y_true, y_pred in [
        ("Train", y_train, y_train_pred),
        ("Test",  y_test,  y_test_pred)]:
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{split:5s} – R²={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

# ── 4. Export predictions ────────────────────────────────────────────────────
pd.DataFrame({'Actual_Train': y_train, 'Predicted_Train': y_train_pred})\
  .to_excel("XGBoost_Train_Predictions.xlsx", index=False)

pd.DataFrame({'Actual_Test': y_test, 'Predicted_Test': y_test_pred})\
  .to_excel("XGBoost_Test_Predictions.xlsx", index=False)

print("\nSaved: XGBoost_Train_Predictions.xlsx, XGBoost_Test_Predictions.xlsx")
