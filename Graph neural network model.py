# Physically- and Knowledge-Informed Deep Learning for Robust Prediction of Martensite Start Temperature in Steels
# By Pengxin Wang and G. M. A. M. El-Fallah
# Contact Dr Gebril El-Fallah:  gmae2@leicester.ac.uk

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Enable saving figures in headless environments; switch to 'TkAgg' for interactive use
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna

from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.loader import DataLoader as GeoDataLoader

# ==================== Reproducibility ====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ==================== I/O Settings ====================
EXCEL_PATH         = '成分物理特征经验公式.xlsx'
MODEL_PATH         = 'best_ms_model_GAT_80_20.pth'
LOSS_CSV           = 'Loss_Curve_GAT_80_20.csv'
LOSS_PNG           = 'Loss_Curve_GAT_80_20.png'
PRED_CSV           = 'Ms_Prediction_TEST_GAT_80_20.csv'
SCATTER_PNG        = 'True_vs_Pred_TEST_GAT_80_20.png'
SCATTER_DATA_CSV   = 'True_vs_Pred_Data_TEST_GAT_80_20.csv'
METRICS_TXT        = 'Metrics_TEST_GAT_80_20.txt'
os.makedirs('.', exist_ok=True)

# ==================== Load & Scale Data ====================
df = pd.read_excel(EXCEL_PATH)

# According to your schema: first 15 composition columns, next 4 physical, next 9 empirical; target column 'Ms'
composition_cols = df.columns[:15]
physical_cols    = df.columns[15:19]
empirical_cols   = df.columns[19:28]
target_col       = 'Ms'

X1 = df[composition_cols].values  # [N, 15]
X2 = df[physical_cols].values     # [N, 4]
X3 = df[empirical_cols].values    # [N, 9]
y  = df[target_col].values.reshape(-1, 1)

# Match your current pipeline: scale each block with StandardScaler
scaler1 = StandardScaler()
scaler2 = StandardScaler()
scaler3 = StandardScaler()
scaler_y = StandardScaler()

X1 = scaler1.fit_transform(X1)
X2 = scaler2.fit_transform(X2)
X3 = scaler3.fit_transform(X3)
y  = scaler_y.fit_transform(y)

# Concatenate into a 28-dimensional feature vector (each dimension will be a node in the graph)
X = np.concatenate([X1, X2, X3], axis=1)  # [N, 28]
N, NUM_NODES = X.shape
assert NUM_NODES == 28, f"Expect 28 features, got {NUM_NODES}"

# ==================== Build Fully Connected Edge Index (without self-loops) ====================
edges_src = []
edges_dst = []
for i in range(NUM_NODES):
    for j in range(NUM_NODES):
        if i != j:
            edges_src.append(i)
            edges_dst.append(j)
edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)  # [2, E]

# ==================== Make Graph List ====================
def create_graph(sample_features, y_scaled_value):
    # Node feature dimension = 1 (one scalar per node)
    x = torch.tensor(sample_features, dtype=torch.float32).unsqueeze(-1)  # [28, 1]
    y_t = torch.tensor([y_scaled_value], dtype=torch.float32)             # [1]
    data = Data(x=x, edge_index=edge_index, y=y_t)
    return data

graphs_all = [create_graph(X[i], y[i, 0]) for i in range(N)]

# ==================== 80/20 Train/Test Split ====================
rng = np.random.default_rng(SEED)
indices = rng.permutation(N)
train_size = int(0.8 * N)
train_idx  = indices[:train_size]
test_idx   = indices[train_size:]

train_graphs = [graphs_all[i] for i in train_idx]
test_graphs  = [graphs_all[i] for i in test_idx]

# Within the training set, carve out 10% as validation for early stopping and tuning (test set untouched)
val_size = max(1, int(0.1 * len(train_graphs)))
true_train_size = len(train_graphs) - val_size
train_graphs_sub = train_graphs[:true_train_size]
val_graphs       = train_graphs[true_train_size:]

def make_loaders(batch_size):
    train_loader = GeoDataLoader(train_graphs_sub, batch_size=batch_size, shuffle=True)
    val_loader   = GeoDataLoader(val_graphs,       batch_size=batch_size, shuffle=False)
    test_loader  = GeoDataLoader(test_graphs,      batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# ==================== GAT Regressor ====================
class GATRegressor(nn.Module):
    def __init__(self, in_channels=1, hidden=32, heads=4, dropout=0.1):
        super().__init__()
        # Two GAT layers; first uses multi-head with concatenation, second uses a single head
        self.gat1 = GATConv(in_channels, hidden, heads=heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden * heads, hidden, heads=1, concat=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.gat2(x, edge_index))
        # Graph-level pooling
        x = global_mean_pool(x, batch)  # [batch_size, hidden]
        out = self.head(x)              # [batch_size, 1]
        return out

# ==================== Train & Eval Utilities ====================
def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred, data.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    losses = []
    for data in loader:
        data = data.to(device)
        pred = model(data)
        loss = criterion(pred, data.y.view(-1, 1))
        losses.append(loss.item())
    return float(np.mean(losses)) if len(losses) > 0 else np.inf

# ==================== Optuna Objective (train/val only) ====================
def objective(trial):
    params = {
        'hidden'    : trial.suggest_categorical('hidden', [16, 32, 48, 64]),
        'heads'     : trial.suggest_categorical('heads',  [2, 4, 8]),
        'dropout'   : trial.suggest_float('dropout', 0.0, 0.5),
        'lr'        : trial.suggest_float('lr', 1e-4, 5e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
    }

    train_loader, val_loader, _ = make_loaders(params['batch_size'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GATRegressor(in_channels=1, hidden=params['hidden'], heads=params['heads'], dropout=params['dropout']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    criterion = nn.MSELoss()

    best_val = np.inf
    patience, trigger = 20, 0

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

study = optuna.create_study(direction='minimize', study_name='GAT_80_20')
study.optimize(objective, n_trials=50)
best_params = study.best_params
print('Best Hyperparameters (GAT):', best_params)

# ==================== Final Training with Best Params (train/val; test used only for final evaluation) ====================
train_loader, val_loader, test_loader = make_loaders(best_params['batch_size'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GATRegressor(in_channels=1,
                     hidden=best_params['hidden'],
                     heads=best_params['heads'],
                     dropout=best_params['dropout']).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
criterion = nn.MSELoss()

train_losses, val_losses = [], []
best_val, patience, trigger = np.inf, 20, 0

for epoch in range(300):
    tr = train_one_epoch(model, train_loader, device, optimizer, criterion)
    va = evaluate(model, val_loader, device, criterion)
    train_losses.append(tr)
    val_losses.append(va)

    if va < best_val:
        best_val = va
        trigger = 0
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        trigger += 1
        if trigger >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# Save loss curves
pd.DataFrame({'Train_Loss': train_losses, 'Val_Loss': val_losses}).to_csv(LOSS_CSV, index=False)
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses,  label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.tight_layout()
plt.savefig(LOSS_PNG, dpi=300)

# ==================== TEST evaluation (test set only) ====================
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

preds_scaled_list = []
trues_scaled_list = []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        pred_scaled = model(data)                 # [B, 1] (scaled)
        preds_scaled_list.append(pred_scaled.cpu().numpy())
        trues_scaled_list.append(data.y.view(-1, 1).cpu().numpy())

pred_scaled = np.vstack(preds_scaled_list)  # (N_test, 1)
true_scaled = np.vstack(trues_scaled_list)  # (N_test, 1)

# Inverse-transform back to original units (°C)
pred = scaler_y.inverse_transform(pred_scaled)
true = scaler_y.inverse_transform(true_scaled)

# Metrics (test set)
rmse = float(np.sqrt(mean_squared_error(true, pred)))
r2   = float(r2_score(true, pred))
mse  = float(mean_squared_error(true, pred))
mae  = float(mean_absolute_error(true, pred))

print('Final Performance on TEST (GAT, 80/20 split):')
print(f'RMSE(Test): {rmse:.3f}')
print(f'R²(Test)  : {r2:.3f}')
print(f'MSE(Test) : {mse:.3f}')
print(f'MAE(Test) : {mae:.3f}')

with open(METRICS_TXT, 'w', encoding='utf-8') as f:
    f.write('Final Performance on TEST (GAT, 80/20 split)\n')
    f.write(f'RMSE(Test): {rmse:.6f}\n')
    f.write(f'R2(Test)  : {r2:.6f}\n')
    f.write(f'MSE(Test) : {mse:.6f}\n')
    f.write(f'MAE(Test) : {mae:.6f}\n')

# Save test-set predictions and scatter data
pd.DataFrame({'True_Ms': true.flatten(), 'Pred_Ms': pred.flatten()}).to_csv(PRED_CSV, index=False)
pd.DataFrame({'True_Ms': true.flatten(), 'Pred_Ms': pred.flatten()}).to_csv(SCATTER_DATA_CSV, index=False)

# Plot test-set scatter
plt.figure()
plt.scatter(true, pred, alpha=0.7)
tmin = min(true.min(), pred.min())
tmax = max(true.max(), pred.max())
plt.plot([tmin, tmax], [tmin, tmax], linestyle='--')
plt.xlabel('True Ms (°C)')
plt.ylabel('Predicted Ms (°C)')
plt.title('True vs Predicted Ms on TEST (GAT, 80/20)')
plt.tight_layout()
plt.savefig(SCATTER_PNG, dpi=300)

print('All done. Files saved:')
print(f'- Best model: {MODEL_PATH}')
print(f'- Loss CSV/PNG: {LOSS_CSV}, {LOSS_PNG}')
print(f'- TEST Predictions CSV: {PRED_CSV}')
print(f'- TEST Scatter PNG: {SCATTER_PNG}')
print(f'- TEST Scatter data CSV: {SCATTER_DATA_CSV}')
print(f'- TEST Metrics TXT: {METRICS_TXT}')
