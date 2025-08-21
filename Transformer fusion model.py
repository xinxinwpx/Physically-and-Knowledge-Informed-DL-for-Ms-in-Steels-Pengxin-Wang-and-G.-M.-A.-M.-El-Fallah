# Physically- and Knowledge-Informed Deep Learning for Robust Prediction of Martensite Start Temperature in Steels
# By Pengxin Wang and G. M. A. M. El-Fallah
# Contact Dr Gebril El-Fallah:  gmae2@leicester.ac.uk

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import random
import matplotlib

matplotlib.use('TkAgg')
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ==================== Load Data ====================
df = pd.read_excel('Dataset.xlsx')

composition_cols = df.columns[:15]
physical_cols = df.columns[15:19]
empirical_cols = df.columns[19:28]
target_col = 'Ms'

X1 = df[composition_cols].values
X2 = df[physical_cols].values
X3 = df[empirical_cols].values
y = df[target_col].values.reshape(-1, 1)

scaler1, scaler2, scaler3, scaler_y = StandardScaler(), StandardScaler(), StandardScaler(), StandardScaler()
X1 = scaler1.fit_transform(X1)
X2 = scaler2.fit_transform(X2)
X3 = scaler3.fit_transform(X3)
y = scaler_y.fit_transform(y)

X1, X2, X3, y = map(lambda x: torch.tensor(x, dtype=torch.float32), (X1, X2, X3, y))
dataset = TensorDataset(X1, X2, X3, y)

# ==================== MLP Branch ====================
def build_mlp(input_dim, depth, hidden_dim, dropout, activation_name):
    act = nn.ReLU() if activation_name == 'ReLU' else nn.LeakyReLU()
    layers = [nn.Linear(input_dim, hidden_dim), act, nn.Dropout(dropout)]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), act, nn.Dropout(dropout)])
    return nn.Sequential(*layers)

# ==================== Transformer Fusion ====================
class TransformerFusion(nn.Module):
    def __init__(self, input_dims, d_model=128, nhead=4):
        super(TransformerFusion, self).__init__()
        self.proj = nn.ModuleList([nn.Linear(dim, d_model) for dim in input_dims])
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.output_dim = d_model

    def forward(self, x_list):
        projected = [proj(x) for proj, x in zip(self.proj, x_list)]
        x_cat = torch.stack(projected, dim=1)  # [B, 3, d_model]
        x_cat = x_cat.permute(1, 0, 2)         # [3, B, d_model]
        encoded = self.encoder(x_cat)
        encoded = encoded.permute(1, 0, 2)     # [B, 3, d_model]
        return encoded.mean(dim=1)

# ==================== Final Model ====================
class MsModelTransformer(nn.Module):
    def __init__(self, params):
        super(MsModelTransformer, self).__init__()
        self.branch1 = build_mlp(15, params['depth1'], params['hidden1'], params['dropout1'], params['act'])
        self.branch2 = build_mlp(4, params['depth2'], params['hidden2'], params['dropout2'], params['act'])
        self.branch3 = build_mlp(9, params['depth3'], params['hidden3'], params['dropout3'], params['act'])
        self.fusion_layer = TransformerFusion([params['hidden1'], params['hidden2'], params['hidden3']],
                                              d_model=params['tf_d_model'], nhead=params['tf_heads'])
        self.head = nn.Sequential(
            nn.Linear(self.fusion_layer.output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x1, x2, x3):
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        out3 = self.branch3(x3)
        fused = self.fusion_layer([out1, out2, out3])
        return self.head(fused)

# ==================== Optuna Hyperparameter Optimization ====================
def objective(trial):
    params = {
        'hidden1': trial.suggest_int('hidden1', 32, 256, step=32),
        'hidden2': trial.suggest_int('hidden2', 16, 128, step=16),
        'hidden3': trial.suggest_int('hidden3', 16, 128, step=16),
        'depth1': trial.suggest_int('depth1', 1, 3),
        'depth2': trial.suggest_int('depth2', 1, 2),
        'depth3': trial.suggest_int('depth3', 1, 2),
        'dropout1': trial.suggest_float('dropout1', 0.05, 0.5),
        'dropout2': trial.suggest_float('dropout2', 0.05, 0.5),
        'dropout3': trial.suggest_float('dropout3', 0.05, 0.5),
        'lr': trial.suggest_float('lr', 1e-4, 5e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64]),
        'act': trial.suggest_categorical('act', ['ReLU', 'LeakyReLU']),
        'tf_d_model': trial.suggest_categorical('tf_d_model', [64, 128, 256]),
        'tf_heads': trial.suggest_categorical('tf_heads', [2, 4, 8]),
    }

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MsModelTransformer(params).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    best_loss, patience, trigger = np.inf, 20, 0

    for epoch in range(200):
        model.train()
        for x1, x2, x3, y_batch in train_loader:
            x1, x2, x3, y_batch = x1.to(device), x2.to(device), x3.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x1, x2, x3), y_batch)
            loss.backward()
            optimizer.step()

        val_loss = sum(criterion(model(x1.to(device), x2.to(device), x3.to(device)), y_batch.to(device)).item()
                       for x1, x2, x3, y_batch in val_loader) / len(val_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            trigger = 0
        else:
            trigger += 1
            if trigger >= patience:
                break
    return best_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
print('Best Hyperparameters:', study.best_params)

# ==================== Final Training with Best Params ====================
best = study.best_params

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=best['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=best['batch_size'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MsModelTransformer(best).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=best['lr'])

train_losses, val_losses = [], []
best_loss, patience, trigger = np.inf, 20, 0

for epoch in range(300):
    model.train()
    running_loss = 0
    for x1, x2, x3, target in train_loader:
        x1, x2, x3, target = x1.to(device), x2.to(device), x3.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x1, x2, x3), target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    val_loss = 0
    model.eval()
    with torch.no_grad():
        for x1, x2, x3, target in val_loader:
            x1, x2, x3, target = x1.to(device), x2.to(device), x3.to(device), target.to(device)
            val_loss += criterion(model(x1, x2, x3), target).item()

    train_losses.append(running_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))

    if val_loss < best_loss:
        best_loss = val_loss
        trigger = 0
        torch.save(model.state_dict(), 'best_ms_model_transformer验证.pth')
    else:
        trigger += 1
        if trigger >= patience:
            print(f"Early Stopping at Epoch {epoch}")
            break

# ==================== Loss Curve Save ====================
pd.DataFrame({'Train_Loss': train_losses, 'Val_Loss': val_losses}).to_csv('Loss_Curve_Transformer验证.csv', index=False)

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Loss_Curve_Transformer.png', dpi=300)
plt.show()

# ==================== Final Test on All Data ====================
model.load_state_dict(torch.load('best_ms_model_transformer.pth'))
model.eval()

X1_all, X2_all, X3_all, y_all = X1.to(device), X2.to(device), X3.to(device), y.to(device)

with torch.no_grad():
    pred = model(X1_all, X2_all, X3_all).cpu().numpy()
    true = y_all.cpu().numpy()

pred = scaler_y.inverse_transform(pred)
true = scaler_y.inverse_transform(true)

# ==================== Evaluation Metrics ====================
rmse = np.sqrt(mean_squared_error(true, pred))
r2 = r2_score(true, pred)
mse = mean_squared_error(true, pred)
mae = mean_absolute_error(true, pred)

print(f'Final Performance:')
print(f'RMSE: {rmse:.3f}')
print(f'R²: {r2:.3f}')
print(f'MSE: {mse:.3f}')
print(f'MAE: {mae:.3f}')

# ==================== Save Prediction Result ====================
pd.DataFrame({'True_Ms': true.flatten(), 'Pred_Ms': pred.flatten()}).to_csv('Ms_Prediction_Result_Transformer验证.csv', index=False)

# ==================== Scatter Plot True vs Pred ====================
plt.scatter(true, pred, alpha=0.7)
plt.xlabel('True Ms')
plt.ylabel('Predicted Ms')
plt.title('True vs Predicted Ms')
plt.plot([true.min(), true.max()], [true.min(), true.max()], 'r--')
plt.savefig('True_vs_Pred_Transformer.png', dpi=300)
plt.show()

pd.DataFrame({'True_Ms': true.flatten(), 'Pred_Ms': pred.flatten()}).to_csv('True_vs_Pred_Data_Transformer.csv', index=False)
