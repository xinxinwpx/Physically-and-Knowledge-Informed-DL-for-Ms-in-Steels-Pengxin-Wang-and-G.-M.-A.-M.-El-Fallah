# Physically- & Knowledge-Informed Models for Martensite Start Temperature (Mₛ) in Steels

A research codebase for predicting martensite start temperature (Mₛ) using classical ML (Lasso, RF/GBDT/MLP/XGBoost), a multi-branch **Transformer Fusion** network, an **Additive-Attention** fusion variant, a **Graph Neural Network (GAT)** baseline, plus **ablation**, **sensitivity**, and **SHAP** interpretability utilities.

> **Authors**: Pengxin Wang; G. M. A. M. El-Fallah  
> **Affiliation**: University of Leicester  
> **Contact**: Dr Gebril El-Fallah — gmae2@leicester.ac.uk

---

## ✨ What’s inside

- **Datasets**  
  - `Dataset.xlsx` (root). Target column typically `Ms` (some scripts also support `Ms(℃)`).
- **Core models**
  - `Transformer fusion model.py` — multi-branch MLP encoders + Transformer fusion + Optuna HPO. :contentReference[oaicite:0]{index=0}
  - `additive-attention fusion model.py` — multi-branch encoders + Bahdanau-style additive attention fusion. :contentReference[oaicite:1]{index=1}
  - `Graph neural network model.py` — GAT on feature-as-node graphs (28 nodes = 15 composition + 4 physical + 9 empirical). :contentReference[oaicite:2]{index=2}
  - `MLP,XGBoost,RF,GBDT.py` — classical baselines with Optuna tuning + plots/exports. :contentReference[oaicite:3]{index=3}
  - `Lasso regression.py` — polynomial (deg=2) Lasso with coefficient sparsity control and readable formula export. :contentReference[oaicite:4]{index=4}
- **Evaluation & analysis**
  - `Transformer fusion model ablation study.py` — branch ablation (No-Comp/No-Phy/No-Emp) metrics. :contentReference[oaicite:5]{index=5}
  - `Transformer fusion model sensitivity analysis.py` — one-at-a-time element sweeps (C, Mn, Si, …, N) with exports. :contentReference[oaicite:6]{index=6}
  - `Transformer fusion model SHAP.py` — global SHAP (KernelExplainer) with publication-style figures. :contentReference[oaicite:7]{index=7}
  - `Transformer fusion model predict.py` — load trained Transformer model and predict on new Excel. :contentReference[oaicite:8]{index=8}

---


**Expected schema of `Dataset.xlsx`:**
- **Composition (15 cols):** C, Mn, Si, Cr, Ni, Mo, V, Co, Al, W, Cu, Nb, Ti, B, N (order matters). :contentReference[oaicite:9]{index=9}  
- **Physical (4 cols):** next four columns after composition (e.g., density/Δr/ΔS_mix etc.; keep consistent across scripts). :contentReference[oaicite:10]{index=10}  
- **Empirical (9 cols):** next nine columns (Ms-related empirical features). :contentReference[oaicite:11]{index=11}  
- **Target:** `Ms` (°C). Some classical scripts also accept `Ms(℃)`. :contentReference[oaicite:12]{index=12}

> If your header differs, adjust the column names or the slices at the top of each script accordingly.

---

## 🚀 Quick start

### 1) Environment
Python ≥ 3.9 is recommended.

# Create & activate env (conda example)
conda create -n ms-pred python=3.10 -y
conda activate ms-pred

# Core deps
pip install numpy pandas scikit-learn matplotlib optuna

# PyTorch (choose your CUDA/CPU build as needed; example CPU):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# XGBoost for classical baselines
pip install xgboost

# SHAP for interpretability
pip install shap

# Optional: torch-geometric (for GNN, pick wheels to match your Torch/CUDA)
# See https://pytorch-geometric.readthedocs.io/en/latest/ for exact commands.
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
If you only need Transformer/Attention and classical baselines, you can skip torch-geometric.

2) Prepare data
Place your dataset at the repo root as Dataset.xlsx. Ensure the column order matches the schema above.

🧪 Training & evaluation
A. Transformer Fusion (main model)
python "Transformer fusion model.py"
What it does:

Standardises data block-wise (15/4/9), runs Optuna HPO, trains with early stopping, saves best weights and loss curves. 

Common outputs (file names may vary by script version):

best_ms_model_transformer.pth, Loss_Curve_Transformer.csv, training/validation loss plot. 

B. Additive-Attention Fusion (variant)
python "additive-attention fusion model.py"
Outputs:

best_ms_model_AdditiveAttn_80_20.pth, loss CSV/PNG, test-set metrics TXT, true-vs-pred CSV/PNG, and branch attention weights CSV per sample. 

C. Classical baselines (MLP / XGBoost / RF / GBDT)
python "MLP,XGBoost,RF,GBDT.py"
Outputs: scatter plots and Excel exports for each model; Optuna-tuned hyperparameters table. 

D. Lasso (sparse polynomial regression)
python "Lasso regression.py"
What it does: builds deg-2 polynomial features, tunes alpha to cap non-zeros (≈≤15), prints a readable Mₛ formula and metrics; supports Ms(℃). 

🔍 Analysis utilities
1) Ablation (which branch matters?)
python "Transformer fusion model ablation study.py"
Loads a trained Transformer model and evaluates No-Composition / No-Physical / No-Empirical variants for RMSE/R²/MAE. 

2) Sensitivity analysis (1-at-a-time sweeps)
python "Transformer fusion model sensitivity analysis.py"
Sweeps each element (C, Mn, …, N; N uses 0–1.8 by default), holding others at mean; exports curves and an Excel summary (e.g., Ms_Sensitivity_15_Transformer.xlsx). 

3) SHAP global importance
python "Transformer fusion model SHAP.py"
Runs KernelExplainer on all features combined; saves publication-style bar figures (Times New Roman), with beautified symbols (e.g., Mₛ). 

4) Predict on new alloys
python "Transformer fusion model predict.py"
Loads best_ms_model_transformer.pth and predicts Mₛ for a new Excel file (edit the placeholder file name in the script). 

5) GNN (GAT) baseline
python "Graph neural network model.py"
Treats each of the 28 features as a node; builds fully connected directed graph (no self-loops); uses GAT + global mean pooling. Saves loss curves and test metrics/plots. 

📊 Notes on features & scaling
Scripts standardise each block separately (composition / physical / empirical) and the target Ms for stable training; predictions are inverse-transformed to °C before reporting metrics. 

If your dataset uses different headers or ordering, adjust the composition_cols / physical_cols / empirical_cols slices at the top of each script. 

✅ Reproducibility tips
Random seeds (e.g., 42) are set in all deep learning scripts; early-stopping patience ≈ 20 epochs; Optuna trials ≈ 50 by default (tune per compute). 

For SHAP, KernelExplainer uses a small background sample (e.g., first 100) to keep runtime reasonable; increase for higher fidelity. 

📜 Citation
If you use this repository, please cite our work (update details as appropriate):

Pengxin Wang, G. M. A. M. El-Fallah. Physically- and Knowledge-Informed Deep Learning for Robust Prediction of Martensite Start Temperature in Steels. 

