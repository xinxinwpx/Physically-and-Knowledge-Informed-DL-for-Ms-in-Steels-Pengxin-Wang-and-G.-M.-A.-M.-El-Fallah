# Physically- and Knowledge-Informed Deep Learning for Robust Prediction of Martensite Start Temperature in Steels
# By Pengxin Wang and G. M. A. M. El-Fallah
# Contact Dr Gebril El-Fallah:  gmae2@leicester.ac.uk

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Read Excel file
file_path = "Dataset.xlsx"
df = pd.read_excel(file_path)

target_column = 'Ms(℃)'
all_features = [col for col in df.columns if col != target_column and col not in ['B', 'N']]

X = df[all_features]
y = df[target_column]

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)
feature_names = poly.get_feature_names_out(all_features)

scaler = StandardScaler()
X_poly_scaled = scaler.fit_transform(X_poly)

X_train, X_test, y_train, y_test = train_test_split(X_poly_scaled, y, test_size=0.2, random_state=42)

def count_nonzero_coefs(alpha):
    model = Lasso(alpha=alpha, max_iter=30000)
    model.fit(X_train, y_train)
    return np.sum(model.coef_ != 0), model

low_alpha, high_alpha = 1e-5, 10
best_alpha = None
best_model = None
best_nonzero = None
max_nonzero_allowed = 15

for _ in range(40):
    mid_alpha = (low_alpha + high_alpha) / 2
    nonzero, model = count_nonzero_coefs(mid_alpha)
    if nonzero <= max_nonzero_allowed:
        best_alpha = mid_alpha
        best_model = model
        best_nonzero = nonzero
        high_alpha = mid_alpha
    else:
        low_alpha = mid_alpha
    if abs(high_alpha - low_alpha) < 1e-7:
        break

if best_model is None:
    best_alpha = high_alpha
    best_model = Lasso(alpha=best_alpha, max_iter=50000)
    best_model.fit(X_train, y_train)
    best_nonzero = np.sum(best_model.coef_ != 0)

print(f"Best alpha: {best_alpha:.8f}")
print(f"Non-zero coefficients: {best_nonzero}")

coefs = best_model.coef_


co_feature_indices = [i for i, f in enumerate(feature_names) if 'Co' in f]
if all(abs(coefs[i]) < 1e-6 for i in co_feature_indices):
    X_co = df[['Co']]
    X_co_train, X_co_test, y_co_train, y_co_test = train_test_split(X_co, y, test_size=0.2, random_state=42)
    linreg_co = LinearRegression()
    linreg_co.fit(X_co_train, y_co_train)
    co_coef = linreg_co.coef_[0]
    try:
        idx_co_linear = feature_names.tolist().index('Co')
        coefs[idx_co_linear] = co_coef
    except ValueError:
        print("Linear term not found; skipping replacement")

# Rescale coefficients and intercept back to original units (de-standardise)
coefs_rescaled = coefs / scaler.scale_
intercept_rescaled = y.mean() - np.dot(coefs_rescaled, scaler.mean_)

# Build human-readable formula
terms = []
for name, coef in zip(feature_names, coefs_rescaled):
    if abs(coef) > 1e-4:
        sign = '+' if coef >= 0 else '-'
        terms.append(f"{sign} {abs(coef):.2f}*{name}")

formula = f"Ms = {intercept_rescaled:.2f} " + " ".join(terms)
print("\nRegression formula:")
print(formula)

# Evaluate on the test set
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# Prediction demo: Alloy A and Alloy B
new_alloys = pd.DataFrame([
    {'C':0.72,'Mn':0.02,'Si':3.87,'Cr':0.01,'Ni':3.4,'Mo':0.21,'V':0,'Co':0.01,'Al':1.39,'W':0,'Cu':0,'Nb':0,'Ti':0},
    {'C':0.45,'Mn':0.15,'Si':0.03,'Cr':0.005,'Ni':13.2,'Mo':0.15,'V':0,'Co':3.99,'Al':2.63,'W':0,'Cu':0,'Nb':0,'Ti':0}
], index=['Alloy A','Alloy B'])

for col in set(all_features)-set(new_alloys.columns):
    new_alloys[col] = 0

new_alloys = new_alloys[all_features]

new_poly = poly.transform(new_alloys)
new_poly_scaled = scaler.transform(new_poly)
ms_pred_new = best_model.predict(new_poly_scaled)

new_alloys['Predicted Ms'] = ms_pred_new
print("\nPredicted Ms for new alloys:")
print(new_alloys[['Predicted Ms']])

# Plot scatter: True vs Predicted (test set)
import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, edgecolors='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('True Ms (℃)')
plt.ylabel('Predicted Ms (℃)')
plt.title('True vs Predicted Ms Scatter Plot')
plt.grid(True)
plt.tight_layout()
plt.show()

# Save test-set prediction results to Excel
results_df = pd.DataFrame({
    'True Ms': y_test,
    'Predicted Ms': y_pred,
    'Residual': y_test - y_pred
})
results_df.to_excel('Ms_Prediction_Results.xlsx', index=False)
print("Test-set prediction results have been saved to Ms_Prediction_Results.xlsx")
