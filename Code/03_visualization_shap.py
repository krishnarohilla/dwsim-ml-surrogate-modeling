import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib
import os

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'Arial', 'font.size': 11})

# 1. Load Data and Train XGBoost
df = pd.read_csv("Dataset.csv")
features = [
    'Feed_Temp_K', 'Feed_Pres_Pa', 'zF_Benzene', 'Feed_Flow_kmolh',
    'N_Stages', 'Feed_Stage', 'Reflux_Ratio', 'Bottoms_Flow_kmolh'
]
targets = ['xD_Benzene', 'xB_Toluene', 'Q_Condenser_kW', 'Q_Reboiler_kW']

X = df[features].values
y = df[targets].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

xgb = XGBRegressor(n_estimators=300, learning_rate=0.04, max_depth=5, random_state=42)
xgb.fit(X_train_scaled, y_train)
y_pred = xgb.predict(X_test_scaled)

# Output directory for plots
os.makedirs("Plots", exist_ok=True)

# -------------------------------------------------------------
# PLOT 1: 4-Target Parity Plots (Predicted vs Actual)
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()
titles = [
    r'Distillate Purity ($x_D$, Benzene)',
    r'Bottoms Purity ($x_B$, Toluene)',
    r'Condenser Duty ($Q_C$, kW)',
    r'Reboiler Duty ($Q_R$, kW)'
]

for i in range(4):
    ax = axes[i]
    ax.scatter(y_test[:, i], y_pred[:, i], color='#1f77b4', alpha=0.7, edgecolors='k', s=35)
    min_v = min(y_test[:, i].min(), y_pred[:, i].min())
    max_v = max(y_test[:, i].max(), y_pred[:, i].max())
    ax.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='Ideal Parity (y=x)')
    ax.set_title(titles[i], fontweight='bold')
    ax.set_xlabel('Actual (DWSIM)')
    ax.set_ylabel('Predicted (XGBoost Surrogate)')
    ax.legend()

plt.tight_layout()
plt.savefig("Plots/01_Parity_Plots.png", dpi=300)
plt.close()
print("Saved: Plots/01_Parity_Plots.png")

# -------------------------------------------------------------
# PLOT 2: Model Comparison Bar Chart
# -------------------------------------------------------------
df_metrics = pd.read_csv("Model_Comparison_Metrics.csv")
# Filter out negative R2 for clean display
df_metrics['R2_clipped'] = df_metrics['R2'].apply(lambda v: max(v, 0.0))

plt.figure(figsize=(10, 5))
sns.barplot(data=df_metrics, x='Target', y='R2_clipped', hue='Model', palette='Blues_r')
plt.title("Model Accuracy Benchmark ($R^2$ Score across Targets)", fontweight='bold')
plt.ylabel(r"$R^2$ Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=15)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig("Plots/02_Model_Comparison_R2.png", dpi=300)
plt.close()
print("Saved: Plots/02_Model_Comparison_R2.png")

# -------------------------------------------------------------
# PLOT 3: Physical Consistency / Monotonicity Check
# -------------------------------------------------------------
reflux_sweep = np.linspace(1.2, 4.5, 50)
base_pt = np.array([350.0, 101325.0, 0.50, 100.0, 20, 10, 2.5, 50.0])
sweep_matrix = np.tile(base_pt, (50, 1))
sweep_matrix[:, 6] = reflux_sweep # Vary Reflux Ratio

sweep_matrix_scaled = scaler.transform(sweep_matrix)
preds_sweep = xgb.predict(sweep_matrix_scaled)

fig, ax1 = plt.subplots(figsize=(8, 5))
color = 'tab:red'
ax1.set_xlabel('Reflux Ratio (RR)', fontweight='bold')
ax1.set_ylabel('Condenser Duty $Q_C$ (kW)', color=color, fontweight='bold')
ax1.plot(reflux_sweep, preds_sweep[:, 2], color=color, lw=2.5, label=r'$Q_C$')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Distillate Purity $x_D$', color=color, fontweight='bold')
ax2.plot(reflux_sweep, preds_sweep[:, 0], color=color, lw=2.5, linestyle='-.', label=r'$x_D$')
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Surrogate Physical Consistency: Trend vs Reflux Ratio", fontweight='bold')
fig.tight_layout()
plt.savefig("Plots/03_Physical_Consistency_Reflux.png", dpi=300)
plt.close()
print("Saved: Plots/03_Physical_Consistency_Reflux.png")

# -------------------------------------------------------------
# PLOT 4: Feature Importance
# -------------------------------------------------------------
feat_imp = pd.DataFrame({
    'Feature': features,
    'Importance': xgb.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(9, 5))
sns.barplot(data=feat_imp, x='Importance', y='Feature', palette='viridis')
plt.title("XGBoost Global Feature Importance", fontweight='bold')
plt.xlabel("Relative Feature Weight")
plt.tight_layout()
plt.savefig("Plots/04_Feature_Importance.png", dpi=300)
plt.close()
print("Saved: Plots/04_Feature_Importance.png")
print("All visual validation plots generated in the /Plots folder.")