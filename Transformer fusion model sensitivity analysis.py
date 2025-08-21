# Physically- and Knowledge-Informed Deep Learning for Robust Prediction of Martensite Start Temperature in Steels
# By Pengxin Wang and G. M. A. M. El-Fallah
# Contact Dr Gebril El-Fallah:  gmae2@leicester.ac.uk

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.use('TkAgg')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# ==================== Define MLP ====================
def build_mlp(input_dim, depth, hidden_dim, dropout, activation_name='ReLU'):
    act = nn.ReLU() if activation_name == 'ReLU' else nn.LeakyReLU()
    layers = [nn.Linear(input_dim, hidden_dim), act, nn.Dropout(dropout)]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), act, nn.Dropout(dropout)]
    return nn.Sequential(*layers)

# ==================== Define Transformer Fusion ====================
class TransformerFusion(nn.Module):
    def __init__(self, input_dims, d_model=64, nhead=8):
        super().__init__()
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

# ==================== Define Final Model ====================
class MsModelTransformer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.branch1 = build_mlp(15, params['depth1'], params['hidden1'], params['dropout1'], params['act'])
        self.branch2 = build_mlp(4, params['depth2'], params['hidden2'], params['dropout2'], params['act'])
        self.branch3 = build_mlp(9, params['depth3'], params['hidden3'], params['dropout3'], params['act'])
        self.fusion = TransformerFusion([params['hidden1'], params['hidden2'], params['hidden3']],
                                        d_model=params['tf_d_model'], nhead=params['tf_heads'])
        self.head = nn.Sequential(nn.Linear(self.fusion.output_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x1, x2, x3):
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        out3 = self.branch3(x3)
        fused = self.fusion([out1, out2, out3])
        return self.head(fused)

# ==================== Load Trained Model ====================
best_params = {
    'hidden1': 192, 'hidden2': 64, 'hidden3': 48,
    'depth1': 2, 'depth2': 1, 'depth3': 1,
    'dropout1': 0.0585, 'dropout2': 0.0942, 'dropout3': 0.2597,
    'lr': 0.0007066, 'batch_size': 64, 'act': 'ReLU',
    'tf_d_model': 64, 'tf_heads': 8
}

model = MsModelTransformer(best_params).to(device)

state_dict = torch.load('best_ms_model_transformer.pth', map_location=device)
new_state_dict = {}
for k, v in state_dict.items():
    if 'fusion_layer' in k:
        new_k = k.replace('fusion_layer', 'fusion')
    else:
        new_k = k
    new_state_dict[new_k] = v

model.load_state_dict(new_state_dict)

model.eval()

# ==================== Sensitivity Analysis ====================
target_elements = ['C', 'Mn', 'Si', 'Cr', 'Ni', 'Mo', 'V', 'Co', 'Al', 'W', 'Cu', 'Nb', 'Ti', 'B', 'N']
colors = plt.cm.tab20.colors
fig, axs = plt.subplots(3, 5, figsize=(25, 12))
axs = axs.flatten()
writer = pd.ExcelWriter('Ms_Sensitivity_15_Transformer.xlsx')

for idx, ele in enumerate(target_elements):
    i = composition_cols.get_loc(ele)
    if ele == 'N':
        test_range = np.linspace(0, 1.8, 100)
    else:
        min_val, max_val = df[ele].min(), df[ele].max()
        test_range = np.linspace(min_val, max_val, 100)

    X_fixed = scaler1.transform(np.tile(df[composition_cols].mean().values, (100, 1)))
    X_fixed[:, i] = (test_range - scaler1.mean_[i]) / scaler1.scale_[i]

    X_total = np.hstack([X_fixed, np.tile(X2.mean(axis=0), (100, 1)), np.tile(X3.mean(axis=0), (100, 1))])
    x_tensor = torch.tensor(X_total, dtype=torch.float32).to(device)
    x1, x2, x3 = x_tensor[:, :15], x_tensor[:, 15:19], x_tensor[:, 19:]

    with torch.no_grad():
        pred = model(x1, x2, x3).cpu().numpy()
    pred = scaler_y.inverse_transform(pred)


    axs[idx].plot(test_range, pred, color=colors[idx % len(colors)], linewidth=3)
    axs[idx].set_xlabel(ele, fontsize=24, fontweight='bold')
    axs[idx].set_ylabel(r'$\mathrm{M}_{\mathrm{s}}$', fontsize=26, fontweight='bold')
    axs[idx].tick_params(labelsize=24, width=3)


    for spine in axs[idx].spines.values():
        spine.set_linewidth(2)

    pd.DataFrame({ele: test_range, 'Ms': pred.flatten()}).to_excel(writer, sheet_name=ele, index=False)

for j in range(len(target_elements), len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout(pad=3.0, w_pad=2.5, h_pad=2.5)
plt.savefig('Ms_Sensitivity_15_Transformer.png', dpi=600, bbox_inches='tight')
plt.savefig('Ms_Sensitivity_15_Transformer.pdf', dpi=600, bbox_inches='tight')
plt.show()
writer.close()


