#Physically- and Knowledge-Informed Deep Learning for Robust Prediction of Martensite Start Temperature in Steels
#By Pengxin Wang and G. M. A. M. El-Fallah
#Contact Dr Gebril El-Fallah:  gmae2@leicester.ac.uk

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import optuna

# Load Excel file
data = pd.read_excel("Dataset.xlsx")

# Extract independent and dependent variables
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Hyperparameter optimization functions
def optimize_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
    }
    model = XGBRegressor(**params)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    return scores.mean()


def optimize_gbdt(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0)
    }
    model = GradientBoostingRegressor(**params)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    return scores.mean()


def optimize_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
    }
    model = RandomForestRegressor(**params)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    return scores.mean()


def optimize_mlp(trial):
    params = {
        'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50, 50), (100, 100)]),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'alpha': trial.suggest_float('alpha', 0.0001, 0.1, log=True),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 0.001, 0.1, log=True),
        'max_iter': trial.suggest_int('max_iter', 200, 1000)
    }
    model = MLPRegressor(**params, random_state=42)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    return scores.mean()


# Perform hyperparameter tuning and individual training
models = {'XGBoost': optimize_xgb, 'GBDT': optimize_gbdt, 'RF': optimize_rf, 'MLP': optimize_mlp}
best_params = {}
predictions = {}
metrics = []

for name, optimize in models.items():
    study = optuna.create_study(direction="maximize")
    study.optimize(optimize, n_trials=50)
    best_params[name] = study.best_params

    if name == 'XGBoost':
        model = XGBRegressor(**study.best_params)
    elif name == 'GBDT':
        model = GradientBoostingRegressor(**study.best_params)
    elif name == 'RF':
        model = RandomForestRegressor(**study.best_params)
    else:
        model = MLPRegressor(**study.best_params, random_state=42)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    predictions[name] = y_pred
    metrics.append([name, mean_squared_error(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred)),
                    mean_absolute_error(y_test, y_pred), r2_score(y_test, y_pred)])

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, label=name, alpha=0.5)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.title(f"Actual vs Predicted Values ({name})")
    plt.savefig(f"{name}_scatter成分物理特征经验公式.png")
    plt.show()

    scatter_df = pd.DataFrame({
        'Actual Values': y_test,
        f'{name} Predictions': y_pred
    })
    scatter_df.to_excel(f"{name}_scatter成分物理特征经验公式.xlsx", index=False)

# Save best hyperparameters
params_df = pd.DataFrame([(name, str(params)) for name, params in best_params.items()],
                         columns=['Model', 'Best Hyperparameters'])
params_df.to_excel("best_hyperparameters成分物理特征经验公式.xlsx", index=False)


print("\n" + "=" * 60)
print("All Models Finished. Results Saved Successfully!")

print("\n" + "=" * 60)
print("Best Hyperparameters of All Models (Detailed):")
for name, params in best_params.items():
    print(f"\n{name} Best Hyperparameters:")
    for key, value in params.items():
        print(f"{key} = {value}")