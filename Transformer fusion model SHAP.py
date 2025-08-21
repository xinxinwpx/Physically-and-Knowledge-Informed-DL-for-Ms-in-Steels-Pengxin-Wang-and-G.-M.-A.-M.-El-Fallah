# Physically- and Knowledge-Informed Deep Learning for Robust Prediction of Martensite Start Temperature in Steels
# By Pengxin Wang and G. M. A. M. El-Fallah
# Contact Dr Gebril El-Fallah:  gmae2@leicester.ac.uk

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.use('TkAgg')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Hyperparameters ====================
best = {
    'hidden1': 192, 'hidden2': 64, 'hidden3': 48,
    'depth1': 2, 'depth2': 1, 'depth3': 1,
    'dropout1': 0.0585, 'dropout2': 0.0942, 'dropout3': 0.2597,
    'lr': 0.0007, 'batch_size': 64, 'act': 'ReLU',
    'tf_d_model': 64, 'tf_heads': 8
}

# ==================== Load Data ====================
df = pd.read_excel('Dataset.xlsx')
df.columns = df.columns.str.strip()  # strip leading/trailing spaces from column names

composition_cols = df.columns[:15]
physical_cols = df.columns[15:19]
empirical_cols = df.columns[19:28]
feature_names = list(composition_cols) + list(physical_cols) + list(empirical_cols)
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

X1, X2, X3, y = map(torch.tensor, (X1, X2, X3, y))
X1_all, X2_all, X3_all, y_all = X1.to(device), X2.to(device), X3.to(device), y.to(device)

# ==================== Feature-name beautifier ====================
def beautify_feature_name(name):
    if name == 'Delta_r':
        return '∆r'
    elif name == 'Delta_S_mix':
        return '∆Sₘᵢₓ'
    elif name.startswith(('MS-', 'Ms-', 'ms-')):
        return name.replace('MS-', 'Mₛ-').replace('Ms-', 'Mₛ-').replace('ms-', 'Mₛ-')
    elif name == 'Ms':
        return 'Mₛ'
    else:
        return name

feature_names = [beautify_feature_name(f) for f in feature_names]  # apply globally

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

# ==================== MsModel Transformer ====================
class MsModelTransformer(nn.Module):
    def __init__(self, params):
        super(MsModelTransformer, self).__init__()
        act = nn.ReLU() if params['act'] == 'ReLU' else nn.LeakyReLU()

        def build_mlp(input_size, hidden, depth, dropout):
            layers = [nn.Linear(input_size, hidden), act, nn.Dropout(dropout)]
            for _ in range(depth - 1):
                layers += [nn.Linear(hidden, hidden), act, nn.Dropout(dropout)]
            return nn.Sequential(*layers)

        self.branch1 = build_mlp(15, params['hidden1'], params['depth1'], params['dropout1'])
        self.branch2 = build_mlp(4, params['hidden2'], params['depth2'], params['dropout2'])
        self.branch3 = build_mlp(9, params['hidden3'], params['depth3'], params['dropout3'])

        self.fusion = TransformerFusion(
            [params['hidden1'], params['hidden2'], params['hidden3']],
            d_model=params['tf_d_model'],
            nhead=params['tf_heads']
        )

        self.head = nn.Sequential(
            nn.Linear(params['tf_d_model'], 64),
            act,
            nn.Linear(64, 1)
        )

    def forward(self, x1, x2, x3):
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        out3 = self.branch3(x3)
        fused = self.fusion([out1, out2, out3])
        return self.head(fused)

# ==================== Load Model ====================
model = MsModelTransformer(best).to(device)
state_dict = torch.load('best_ms_model_transformer.pth', map_location=device)
new_state_dict = {k.replace('fusion_layer', 'fusion'): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model.eval()

# ==================== Global SHAP analysis (using all samples) ====================
X_all = torch.cat([X1_all, X2_all, X3_all], dim=1).cpu().numpy()

class WrappedModel:
    def __init__(self, model):
        self.model = model

    def __call__(self, x_numpy):
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32).to(device)
        x1, x2, x3 = x_tensor[:, :15], x_tensor[:, 15:19], x_tensor[:, 19:]
        with torch.no_grad():
            pred = self.model(x1, x2, x3)
        return pred.cpu().numpy()

wrapped_model = WrappedModel(model)

print("Starting SHAP value computation (all samples)...")
explainer = shap.KernelExplainer(wrapped_model, X_all[:100])  # first 100 as background
shap_values = explainer.shap_values(X_all)
print("SHAP value computation finished!")

# ==================== Feature Importance Bar Plot ====================
shap_sum = np.abs(shap_values[0]).mean(axis=0)
importance_df = pd.DataFrame({'Feature': feature_names, 'Mean_SHAP': shap_sum})
importance_df = importance_df.sort_values(by='Mean_SHAP', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Mean_SHAP'])
plt.xlabel('Mean |SHAP Value|', fontsize=18, fontname='Times New Roman')
plt.title('Feature Importance (All Features Together)', fontsize=20, fontname='Times New Roman')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('Feature_Importance_Transformer_Overall.png', dpi=300)
plt.show()

importance_df.to_csv('Feature_Importance_Transformer_Overall.csv', index=False)

# ==================== SHAP Beeswarm Plot ====================
plt.figure()
shap.summary_plot(
    shap_values[0],
    X_all,
    feature_names=feature_names,
    show=False,
    plot_type='dot',
    max_display=len(feature_names)
)

fig = plt.gcf()
ax = plt.gca()
ax.set_xlabel('SHAP value (impact on model output)', fontsize=24, fontname='Times New Roman')
ax.set_ylabel('Features', fontsize=24, fontname='Times New Roman')
ax.set_title('SHAP Beeswarm Plot (All Features)', fontsize=26, fontname='Times New Roman')
ax.tick_params(axis='both', labelsize=20)

plt.tight_layout()
plt.savefig('SHAP_Beeswarm_Transformer_Overall_AllFeatures.png', dpi=600, bbox_inches='tight')
plt.savefig('SHAP_Beeswarm_Transformer_Overall_AllFeatures.pdf', dpi=600, bbox_inches='tight')
plt.show()

print("✅ Global SHAP analysis completed—figures and CSV have been saved!")
