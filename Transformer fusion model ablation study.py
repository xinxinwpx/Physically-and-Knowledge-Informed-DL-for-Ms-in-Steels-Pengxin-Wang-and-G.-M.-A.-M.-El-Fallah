# Physically- and Knowledge-Informed Deep Learning for Robust Prediction of Martensite Start Temperature in Steels
# By Pengxin Wang and G. M. A. M. El-Fallah
# Contact Dr Gebril El-Fallah:  gmae2@leicester.ac.uk

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

# ==================== Load Dataset ====================
df = pd.read_excel('Dataset.xlsx')
X1 = df.iloc[:, :15].values  # Composition
X2 = df.iloc[:, 15:19].values  # Physical
X3 = df.iloc[:, 19:28].values  # Empirical
y = df['Ms'].values.reshape(-1, 1)

scaler1, scaler2, scaler3, scaler_y = StandardScaler(), StandardScaler(), StandardScaler(), StandardScaler()
X1 = scaler1.fit_transform(X1)
X2 = scaler2.fit_transform(X2)
X3 = scaler3.fit_transform(X3)
y = scaler_y.fit_transform(y)

X1 = torch.tensor(X1, dtype=torch.float32)
X2 = torch.tensor(X2, dtype=torch.float32)
X3 = torch.tensor(X3, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
dataset = torch.utils.data.TensorDataset(X1, X2, X3, y)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=64)

# ==================== Best Hyperparameters ====================
best = {
    'hidden1': 192, 'hidden2': 64, 'hidden3': 48,
    'depth1': 2, 'depth2': 1, 'depth3': 1,
    'dropout1': 0.0585, 'dropout2': 0.0942, 'dropout3': 0.2597,
    'lr': 0.0007, 'batch_size': 64, 'act': 'ReLU',
    'tf_d_model': 64, 'tf_heads': 8
}

# ==================== Model Definition ====================
def build_mlp(input_dim, depth, hidden_dim, dropout, activation_name):
    act = nn.ReLU() if activation_name == 'ReLU' else nn.LeakyReLU()
    layers = [nn.Linear(input_dim, hidden_dim), act, nn.Dropout(dropout)]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), act, nn.Dropout(dropout)])
    return nn.Sequential(*layers)

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

class MsModelTransformer(nn.Module):
    def __init__(self, params):
        super(MsModelTransformer, self).__init__()
        self.branch1 = build_mlp(15, params['depth1'], params['hidden1'], params['dropout1'], params['act'])
        self.branch2 = build_mlp(4, params['depth2'], params['hidden2'], params['dropout2'], params['act'])
        self.branch3 = build_mlp(9, params['depth3'], params['hidden3'], params['dropout3'], params['act'])
        self.fusion_layer = TransformerFusion([params['hidden1'], params['hidden2'], params['hidden3']],
                                              d_model=params['tf_d_model'], nhead=params['tf_heads'])
        self.head = nn.Sequential(nn.Linear(self.fusion_layer.output_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x1, x2, x3):
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        out3 = self.branch3(x3)
        fused = self.fusion_layer([out1, out2, out3])
        return self.head(fused)

# ==================== Load Pretrained Model ====================
model = MsModelTransformer(best).to(device)
model.load_state_dict(torch.load('best_ms_model_transformer.pth'))
model.eval()

# ==================== Ablation Wrapper ====================
class MsTransformerAblation(nn.Module):
    def __init__(self, base_model, disable_branch):
        super(MsTransformerAblation, self).__init__()
        self.base = base_model
        self.disable = disable_branch

    def forward(self, x1, x2, x3):
        if self.disable == 1: x1 = torch.zeros_like(x1)
        if self.disable == 2: x2 = torch.zeros_like(x2)
        if self.disable == 3: x3 = torch.zeros_like(x3)
        return self.base(x1, x2, x3)

# ==================== Evaluation Function ====================
def evaluate(model):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x1, x2, x3, y in val_loader:
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
            out = model(x1, x2, x3)
            preds.append(out.cpu().numpy())
            trues.append(y.cpu().numpy())
    preds, trues = np.vstack(preds), np.vstack(trues)
    preds, trues = scaler_y.inverse_transform(preds), scaler_y.inverse_transform(trues)
    rmse = np.sqrt(mean_squared_error(trues, preds))
    r2 = r2_score(trues, preds)
    mae = mean_absolute_error(trues, preds)
    return rmse, r2, mae

# ==================== Run Ablation ====================
rmse_full, _, _ = evaluate(model)
rmse_no1, _, _ = evaluate(MsTransformerAblation(model, 1))  # No Composition
rmse_no2, _, _ = evaluate(MsTransformerAblation(model, 2))  # No Physical
rmse_no3, _, _ = evaluate(MsTransformerAblation(model, 3))  # No Empirical

# ==================== Plot and Save Results ====================
plt.figure(figsize=(6, 4))
plt.bar(['Full', 'No-Comp', 'No-Phy', 'No-Emp'], [rmse_full, rmse_no1, rmse_no2, rmse_no3], color='skyblue')
plt.ylabel('RMSE')
plt.title('Ablation Study on Transformer Fusion Model')
plt.tight_layout()
plt.savefig('Ablation_RMSE_Transformer.png', dpi=300)
plt.show()

pd.DataFrame({
    'Mode': ['Full', 'No-Comp', 'No-Phy', 'No-Emp'],
    'RMSE': [rmse_full, rmse_no1, rmse_no2, rmse_no3]
}).to_csv('Ablation_RMSE_Transformer.csv', index=False)

print('✅ Ablation test completed and saved.')
