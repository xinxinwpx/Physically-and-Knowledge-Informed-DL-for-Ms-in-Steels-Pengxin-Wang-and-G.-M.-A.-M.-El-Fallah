#Physically- and Knowledge-Informed Deep Learning for Robust Prediction of Martensite Start Temperature in Steels
#By Pengxin Wang and G. M. A. M. El-Fallah
#Contact Dr Gebril El-Fallah:  gmae2@leicester.ac.uk


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ============= Load Training Data for Scalers =============
df_train = pd.read_excel('Dataset.xlsx')
composition_cols = df_train.columns[:15]
physical_cols = df_train.columns[15:19]
empirical_cols = df_train.columns[19:28]
target_col = 'Ms'

X1_train = df_train[composition_cols].values
X2_train = df_train[physical_cols].values
X3_train = df_train[empirical_cols].values
y_train = df_train[target_col].values.reshape(-1, 1)


scaler1, scaler2, scaler3, scaler_y = StandardScaler(), StandardScaler(), StandardScaler(), StandardScaler()
X1_train = scaler1.fit_transform(X1_train)
X2_train = scaler2.fit_transform(X2_train)
X3_train = scaler3.fit_transform(X3_train)
scaler_y.fit(y_train)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============= Best Hyperparameters =============
best = {
    'hidden1': 192, 'hidden2': 64, 'hidden3': 48,
    'depth1': 2, 'depth2': 1, 'depth3': 1,
    'dropout1': 0.0585, 'dropout2': 0.0942, 'dropout3': 0.2597,
    'lr': 0.0007, 'batch_size': 64, 'act': 'ReLU',
    'tf_d_model': 64, 'tf_heads': 8
}

# ============= Define Model Structure =============
def build_mlp(input_dim, depth, hidden_dim, dropout, act_fn):
    layers = [nn.Linear(input_dim, hidden_dim), act_fn, nn.Dropout(dropout)]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), act_fn, nn.Dropout(dropout)]
    return nn.Sequential(*layers)

class TransformerFusion(nn.Module):
    def __init__(self, input_dims, d_model, nhead):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(dim, d_model) for dim in input_dims])
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x_list):
        projected = [proj(x) for proj, x in zip(self.proj, x_list)]
        x_cat = torch.stack(projected, dim=1).permute(1, 0, 2)
        encoded = self.encoder(x_cat).permute(1, 0, 2)
        return encoded.mean(dim=1)

class MsModelTransformer(nn.Module):
    def __init__(self, params):
        super().__init__()
        act = nn.ReLU() if params['act'] == 'ReLU' else nn.LeakyReLU()
        self.branch1 = build_mlp(15, params['depth1'], params['hidden1'], params['dropout1'], act)
        self.branch2 = build_mlp(4, params['depth2'], params['hidden2'], params['dropout2'], act)
        self.branch3 = build_mlp(9, params['depth3'], params['hidden3'], params['dropout3'], act)
        self.fusion = TransformerFusion([params['hidden1'], params['hidden2'], params['hidden3']],
                                        d_model=params['tf_d_model'], nhead=params['tf_heads'])
        self.head = nn.Sequential(nn.Linear(params['tf_d_model'], 64), act, nn.Linear(64, 1))

    def forward(self, x1, x2, x3):
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        out3 = self.branch3(x3)
        fused = self.fusion([out1, out2, out3])
        return self.head(fused)

# ============= Load Trained Model =============
model = MsModelTransformer(best).to(device)
state_dict = torch.load('best_ms_model_transformer.pth', map_location=device)

# 替换参数名字 fusion_layer -> fusion
state_dict = {k.replace('fusion_layer', 'fusion'): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

# ============= Load New Data =============
df_new = pd.read_excel('file name.xlsx')  #
X1_new = scaler1.transform(df_new[composition_cols].values)
X2_new = scaler2.transform(df_new[physical_cols].values)
X3_new = scaler3.transform(df_new[empirical_cols].values)

# ============= Predict New Data =============
with torch.no_grad():
    X1_tensor = torch.tensor(X1_new).float().to(device)
    X2_tensor = torch.tensor(X2_new).float().to(device)
    X3_tensor = torch.tensor(X3_new).float().to(device)
    preds_scaled = model(X1_tensor, X2_tensor, X3_tensor).cpu().numpy()
    preds = scaler_y.inverse_transform(preds_scaled).flatten()

# ============= Save Results =============
df_new['Predicted_Ms'] = preds
df_new.to_excel('result.xlsx', index=False)


