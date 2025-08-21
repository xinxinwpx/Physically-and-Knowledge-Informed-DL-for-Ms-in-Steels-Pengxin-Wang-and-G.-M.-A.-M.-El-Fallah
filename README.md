# Physically- & Knowledge-Informed Models for Martensite Start Temperature (Mₛ) in Steels

## Overview
Code and curated data for predicting martensite start temperature (Mₛ) in steels. The repository provides: (i) classical ML baselines (Lasso, RF/GBDT/MLP/XGBoost); (ii) a multi-branch Transformer Fusion model; (iii) an Additive-Attention fusion variant; (iv) a Graph Neural Network (GAT) baseline; plus utilities for ablation, sensitivity analysis, SHAP interpretability, and batch prediction.  
**Authors:** Pengxin Wang; G. M. A. M. El-Fallah (University of Leicester).  
**Contact:** gmae2@leicester.ac.uk

## Data format
Place `Dataset.xlsx` at the repository root. Column order matters: Composition (15) = C, Mn, Si, Cr, Ni, Mo, V, Co, Al, W, Cu, Nb, Ti, B, N; Physical (4) = the next four descriptors used in the scripts; Empirical (9) = the next nine Mₛ-related formula features; Target = `Ms` (°C). If your headers differ, edit the column selections at the top of the scripts accordingly.

## Installation (Python ≥ 3.9)
- (optional) create env: conda create -n ms-pred python=3.10 -y && conda activate ms-pred  
- basics: pip install numpy pandas scikit-learn matplotlib optuna shap xgboost  
- PyTorch (choose build as needed; example CPU): pip install torch --index-url https://download.pytorch.org/whl/cpu  
- (optional) PyTorch Geometric for the GNN baseline (pick wheels matching your Torch/CUDA): pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cpu.html

## How to run (key scripts)
- Transformer Fusion (main model): python "Transformer fusion model.py"  
- Additive-Attention Fusion (variant): python "additive-attention fusion model.py"  
- Classical baselines (MLP, XGBoost, RF, GBDT): python "MLP,XGBoost,RF,GBDT.py"  
- Lasso (sparse polynomial regression with readable formula): python "Lasso regression.py"  
- Graph Neural Network (GAT) baseline: python "Graph neural network model.py"  
- Ablation (No-Comp / No-Phy / No-Emp): python "Transformer fusion model ablation study.py"  
- Sensitivity analysis (one-at-a-time element sweeps): python "Transformer fusion model sensitivity analysis.py"  
- SHAP global feature importance: python "Transformer fusion model SHAP.py"  
- Batch prediction with a trained Transformer model: python "Transformer fusion model predict.py"

## Outputs
Scripts save (names may vary): best model weights (e.g., `best_ms_model_transformer.pth` or `best_ms_model_AdditiveAttn_80_20.pth`), loss curves (CSV/PNG), test metrics (RMSE/MAE/R²), true-vs-pred scatter (CSV/PNG), ablation tables, sensitivity CSVs/figures, and SHAP figures (publication-style with Times New Roman).

## Reproducibility notes
Fixed random seeds (e.g., 42) are set; data are standardised per branch (composition/physical/empirical); early stopping is enabled; Optuna trials default to a moderate value (tune for your compute). Predictions are inverse-transformed to °C before reporting metrics.

## Citation
If you use this repository, please cite: Pengxin Wang; G. M. A. M. El-Fallah. *Physically- and Knowledge-Informed Deep Learning for Robust Prediction of Martensite Start Temperature in Steels*, University of Leicester, 2025.

## Licence
Add a licence file of your choice (e.g., MIT or Apache-2.0).

## Acknowledgements
We thank colleagues for discussions. This work uses PyTorch, scikit-learn, XGBoost, SHAP, Optuna, and (optionally) PyTorch Geometric.
