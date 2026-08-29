import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# Load dataset
df = pd.read_csv("Dataset.csv")

features = [
    'Feed_Temp_K', 'Feed_Pres_Pa', 'zF_Benzene', 'Feed_Flow_kmolh',
    'N_Stages', 'Feed_Stage', 'Reflux_Ratio', 'Bottoms_Flow_kmolh'
]
targets = ['xD_Benzene', 'xB_Toluene', 'Q_Condenser_kW', 'Q_Reboiler_kW']

X = df[features].values
y = df[targets].values

# 80/20 Train-Test split[cite: 3]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

models = {
    "Polynomial Ridge": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.04, max_depth=5, random_state=42),
    "Artificial Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=800, random_state=42)
}

summary_rows = []

print("=" * 80)
print("             SURROGATE MODEL BENCHMARKING (BENZENE-TOLUENE COLUMN)")
print("=" * 80)

best_model_obj = None

for name, model in models.items():
    if name == "Polynomial Ridge":
        model.fit(X_train_poly, y_train)
        y_pred = model.predict(X_test_poly)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
    if name == "XGBoost":
        best_model_obj = model
        
    print(f"\n>>> Model: {name}")
    for i, target in enumerate(targets):
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        print(f"  {target:18s} | R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
        summary_rows.append({
            "Model": name, "Target": target, "R2": round(r2, 4), 
            "RMSE": round(rmse, 4), "MAE": round(mae, 4)
        })

# Export benchmark comparison summary[cite: 3]
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv("Model_Comparison_Metrics.csv", index=False)

# Save best model artifacts
joblib.dump(best_model_obj, "best_surrogate_xgboost.pkl")
joblib.dump(scaler_X, "scaler_X.pkl")
print("\n" + "=" * 80)
print("Benchmarking completed. Summary saved to Model_Comparison_Metrics.csv")