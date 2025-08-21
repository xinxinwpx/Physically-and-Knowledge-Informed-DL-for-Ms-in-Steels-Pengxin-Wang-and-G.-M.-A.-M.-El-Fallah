# Physically- and Knowledge-Informed Deep Learning for Robust Prediction of Martensite Start Temperature in Steels
# By Pengxin Wang and G. M. A. M. El-Fallah
# Contact Dr Gebril El-Fallah:  gmae2@leicester.ac.uk

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib
matplotlib.use('Agg')  # Allow saving figures in headless environments; switch to 'TkAgg' for interactive use
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna

# ==================== Reproducibility ====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ==================== I/O Settings ====================
EXCEL_PATH = '成分物理特征经验公式.xlsx'
MODEL_PATH = 'best_ms_model_AdditiveAttn_80_20.pth'
LOSS_CSV   = 'Loss_Curve_AdditiveAttn_80_20.csv'
LOSS_PNG   = 'Loss_Curve_AdditiveAttn_80_20.png'
PRED_CSV   = 'Ms_Prediction_TEST_AdditiveAttn_80_20.csv'
SCATTER_PNG= 'True_vs_Pred_TEST_AdditiveAttn_80_20.png'
SCATTER_DATA_CSV = 'True_vs_Pred_Data_TEST_AdditiveAttn_80_20.csv'
ATTN_CSV   = 'Branch_Attention_Weights_TEST_AdditiveAttn_80_20.csv'
METRICS_TXT= 'Metrics_TEST_AdditiveAttn_80_20.txt'

os.makedirs('.', exist_ok=True)

# ==================== Load Data ====================
df = pd.read_excel(EXCEL_PATH)

# Split columns according to your data schema
composition_cols = df.columns[:15]
physical_cols    = df.columns[15:19]
empirical_cols   = df.columns[19:28]
target_col       = 'Ms'

X1 = df[composition_cols].values  # [N, 15]
X2 = df[physical_cols].values     # [N, 4]
X3 = df[empirical_cols].values    # [N, 9]
y  = df[target_col].values.reshape(-1, 1)

# Standardisation (replace with min–max if preferred)
scaler1 = StandardScaler()
scaler2 = StandardScaler()
scaler3 = StandardScaler()
scaler_y = StandardScaler()

X1 = scaler1.fit_transform(X1)
X2 = scaler2.fit_transform(X2)
X3 = scaler3.fit_transform(X3)
y  = scaler_y.fit_transform(y)

# To tensors
X1 = torch.tensor(X1, dtype=torch.float32)
X2 = torch.tensor(X2, dtype=torch.float32)
X3 = torch.tensor(X3, dtype=torch.float32)
y  = torch.tensor(y,  dtype=torch.float32)

full_dataset = TensorDataset(X1, X2, X3, y)

# ==================== Fixed 80/20 Train/Test Split ====================
N = len(full_dataset)
train_size = int(0.8 * N)
test_size  = N - train_size
train_all_ds, test_ds = random_split(
    full_dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(SEED)
)

# For early stopping and tuning, hold out a small validation portion from the training set (test set untouched)
val_portion = max(1, int(0.1 * train_size))  # 10% of the training set for validation
true_train_size = train_size - val_portion
train_ds, val_ds = random_split(
    train_all_ds,
    [true_train_size, val_portion],
    generator=torch.Generator().manual_seed(SEED)
)

def make_loaders(batch_size):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# ==================== MLP Branch Builder ====================
def build_mlp(input_dim, depth, hidden_dim, dropout, activation_name='ReLU'):
    act = nn.ReLU() if activation_name == 'ReLU' else nn.LeakyReLU()
    layers = [nn.Linear(input_dim, hidden_dim), act, nn.Dropout(dropout)]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), act, nn.Dropout(dropout)]
    return nn.Sequential(*layers)

# ==================== Additive Attention Fusion ====================
class AdditiveAttentionFusion(nn.Module):
    """
    Three-branch additive (Bahdanau-style) attention:
    score_i = v^T tanh(W h_i + b), apply softmax over i ∈ {1,2,3}, then take the weighted sum.
    """
    def __init__(self, input_dims, d_model=128):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(dim, d_model) for dim in input_dims])
        self.W = nn.Linear(d_model, d_model, bias=True)
        self.v = nn.Linear(d_model, 1, bias=False)
        self.output_dim = d_model

    def forward(self, x_list):
        # x_list: [h1, h2, h3], each with shape [B, d_i]
        H = [proj(x) for proj, x in zip(self.proj, x_list)]  # list of [B, d_model]
        H = torch.stack(H, dim=1)                             # [B, 3, d_model]
        scores = self.v(torch.tanh(self.W(H)))                # [B, 3, 1]
        attn = torch.softmax(scores, dim=1)                   # [B, 3, 1]
        fused = (attn * H).sum(dim=1)                         # [B, d_model]
        return fused, attn.squeeze(-1)                        # [B, 3]

# ==================== Final Model (Additive Attention) ====================
class MsModelAdditiveAttn(nn.Module):
    def __init__(self, params):
        super().__init__()
        act = params.get('act', 'ReLU')
        self.branch1 = build_mlp(15, params['depth1'], params['hidden1'], params['dropout1'], act)
        self.branch2 = build_mlp(4,  params['depth2'], params['hidden2'], params['dropout2'], act)
        self.branch3 = build_mlp(9,  params['depth3'], params['hidden3'], params['dropout3'], act)
        self.fusion  = AdditiveAttentionFusion(
            [params['hidden1'], params['hidden2'], params['hidden3']],
            d_model=params['d_model']
        )
        self.head = nn.Sequential(
            nn.Linear(self.fusion.output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x1, x2, x3, return_attn=False):
        h1 = self.branch1(x1)
        h2 = self.branch2(x2)
        h3 = self.branch3(x3)
        fused, attn = self.fusion([h1, h2, h3])
        out = self.head(fused)
        if return_attn:
            return out, attn
        return out

# ==================== Train & Eval Utilities ====================
def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total = 0.0
    for x1, x2, x3, yb in loader:
        x1, x2, x3, yb = x1.to(device), x2.to(device), x3.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(x1, x2, x3)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    losses = []
    for x1, x2, x3, yb in loader:
        x1, x2, x3, yb = x1.to(device), x2.to(device), x3.to(device), yb.to(device)
        pred = model(x1, x2, x3)
        losses.append(criterion(pred, yb).item())
    return float(np.mean(losses))

# ==================== Optuna Objective (train/val only) ====================
def objective_additive(trial):
    params = {
        'hidden1' : trial.suggest_int('hidden1', 32, 256, step=32),
        'hidden2' : trial.suggest_int('hidden2', 16, 128, step=16),
        'hidden3' : trial.suggest_int('hidden3', 16, 128, step=16),
        'depth1'  : trial.suggest_int('depth1', 1, 3),
        'depth2'  : trial.suggest_int('depth2', 1, 2),
        'depth3'  : trial.suggest_int('depth3', 1, 2),
        'dropout1': trial.suggest_float('dropout1', 0.05, 0.5),
        'dropout2': trial.suggest_float('dropout2', 0.05, 0.5),
        'dropout3': trial.suggest_float('dropout3', 0.05, 0.5),
        'd_model' : trial.suggest_categorical('d_model', [64, 96, 128, 192]),
        'lr'      : trial.suggest_float('lr', 1e-4, 5e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64]),
        'act'     : trial.suggest_categorical('act', ['ReLU', 'LeakyReLU']),
    }

    train_loader, val_loader, _ = make_loaders(params['batch_size'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MsModelAdditiveAttn(params).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    best_val = np.inf
    patience = 20
    trigger  = 0

    for epoch in range(200):
        _ = train_one_epoch(model, train_loader, device, optimizer, criterion)
        val_loss = evaluate(model, val_loader, device, criterion)

        if val_loss < best_val:
            best_val = val_loss
            trigger = 0
        else:
            trigger += 1
            if trigger >= patience:
                break

    return best_val

# ==================== Hyperparameter Search ====================
study = optuna.create_study(direction='minimize', study_name='AdditiveAttentionStudy_80_20')
study.optimize(objective_additive, n_trials=50)
print('Best Hyperparameters (Additive-Attn):', study.best_params)

# ==================== Final Training with Best Params (still train/val only; test used only for final evaluation) ====================
best = study.best_params
train_loader, val_loader, test_loader = make_loaders(best['batch_size'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MsModelAdditiveAttn(best).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=best['lr'])

train_losses, val_losses = [], []
best_loss, patience, trigger = np.inf, 20, 0

for epoch in range(300):
    tr_loss = train_one_epoch(model, train_loader, device, optimizer, criterion)
    va_loss = evaluate(model, val_loader, device, criterion)
    train_losses.append(tr_loss)
    val_losses.append(va_loss)

    if va_loss < best_loss:
        best_loss = va_loss
        trigger = 0
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        trigger += 1
        if trigger >= patience:
            print(f"Early Stopping at Epoch {epoch}")
            break

# Save loss curves (train/val)
pd.DataFrame({'Train_Loss': train_losses, 'Val_Loss': val_losses}).to_csv(LOSS_CSV, index=False)
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses,  label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(LOSS_PNG, dpi=300)

# ==================== Final evaluation and plotting on the TEST set only ====================
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

all_true_scaled, all_pred_scaled, all_attn = [], [], []
with torch.no_grad():
    for x1, x2, x3, yb in test_loader:
        x1, x2, x3 = x1.to(device), x2.to(device), x3.to(device)
        pred_scaled, attn_w = model(x1, x2, x3, return_attn=True)
        all_pred_scaled.append(pred_scaled.cpu().numpy())
        all_true_scaled.append(yb.numpy())
        all_attn.append(attn_w.cpu().numpy())

pred_scaled = np.vstack(all_pred_scaled)  # (N_test, 1)
true_scaled = np.vstack(all_true_scaled)  # (N_test, 1)
attn_weights = np.vstack(all_attn)        # (N_test, 3)

# Inverse scaling (TEST only)
pred = scaler_y.inverse_transform(pred_scaled)
true = scaler_y.inverse_transform(true_scaled)

# Metrics (TEST only)
rmse = float(np.sqrt(mean_squared_error(true, pred)))
r2   = float(r2_score(true, pred))
mse  = float(mean_squared_error(true, pred))
mae  = float(mean_absolute_error(true, pred))

print('Final Performance on TEST (Additive Attention, 80/20 split):')
print(f'RMSE(Test): {rmse:.3f}')
print(f'R²(Test)  : {r2:.3f}')
print(f'MSE(Test) : {mse:.3f}')
print(f'MAE(Test) : {mae:.3f}')

with open(METRICS_TXT, 'w', encoding='utf-8') as f:
    f.write('Final Performance on TEST (Additive Attention, 80/20 split)\n')
    f.write(f'RMSE(Test): {rmse:.6f}\n')
    f.write(f'R2(Test)  : {r2:.6f}\n')
    f.write(f'MSE(Test) : {mse:.6f}\n')
    f.write(f'MAE(Test) : {mae:.6f}\n')

# Save TEST predictions
pd.DataFrame({'True_Ms': true.flatten(), 'Pred_Ms': pred.flatten()}).to_csv(PRED_CSV, index=False)
pd.DataFrame({'True_Ms': true.flatten(), 'Pred_Ms': pred.flatten()}).to_csv(SCATTER_DATA_CSV, index=False)

# Save per-sample TEST attention weights with summary statistics at the top
attn_df = pd.DataFrame(attn_weights, columns=['Branch_Composition', 'Branch_Physical', 'Branch_Empirical'])
attn_stats = attn_df.agg(['mean', 'std'])
with open(ATTN_CSV, 'w', encoding='utf-8') as f:
    f.write('# TEST Attention Weights Statistics (mean/std)\n')
    attn_stats.to_csv(f)
    f.write('\n# TEST Per-sample Attention Weights\n')
attn_df.to_csv(ATTN_CSV, mode='a', index=False)

# Plot TEST True vs Predicted scatter
plt.figure()
plt.scatter(true, pred, alpha=0.7)
_min = min(true.min(), pred.min())
_max = max(true.max(), pred.max())
plt.plot([_min, _max], [_min, _max], linestyle='--')
plt.xlabel('True Ms (°C)')
plt.ylabel('Predicted Ms (°C)')
plt.title('True vs Predicted Ms on TEST (Additive Attention, 80/20)')
plt.tight_layout()
plt.savefig(SCATTER_PNG, dpi=300)

print('All done. Files saved:')
print(f'- Best model: {MODEL_PATH}')
print(f'- Loss CSV/PNG: {LOSS_CSV}, {LOSS_PNG}')
print(f'- TEST Predictions CSV: {PRED_CSV}')
print(f'- TEST Scatter PNG: {SCATTER_PNG}')
print(f'- TEST Scatter data CSV: {SCATTER_DATA_CSV}')
print(f'- TEST Attention weights CSV: {ATTN_CSV}')
print(f'- TEST Metrics TXT: {METRICS_TXT}')
