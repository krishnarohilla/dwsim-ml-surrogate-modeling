# Surrogate Modeling of Binary Distillation Column

## Overview
This repository provides a machine learning surrogate framework for a Benzene-Toluene distillation column modeled with the Peng-Robinson equation of state in DWSIM.

## Folder Structure
- `Code/01_data_generation.py`: Generates Latin Hypercube samples and runs column evaluation.
- `Code/02_model_training_comparison.py`: Trains and benchmarks 4 ML models (Ridge, Random Forest, XGBoost, MLP).
- `Code/03_visualization_shap.py`: Generates parity plots, feature importances, and monotonicity checks.
- `Dataset.csv`: Sampled simulation dataset (500 operating points).
- `Plots/`: Generated visual figures and parity charts.
- `benzene_toluene_column.dwxml`: Base DWSIM simulation flowsheet file.
- `Results_Summary.txt`: Core metrics and sample prediction verification.

## Instructions to Run
1. Install requirements:
   ```bash
   pip install numpy pandas scipy scikit-learn xgboost matplotlib seaborn shap pythonnet
