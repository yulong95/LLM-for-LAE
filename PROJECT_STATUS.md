# Project Status

## Official GPT-2 Model
- Checkpoint: `output/GPT2_08.31_12-13-10/best.bin`
- Training: gamma=0.4, gamma2=5.0, 100 epochs, lr=1e-4
- Data split: 80/10/10 (train/val/test)

## Official CNN Model
- Checkpoint: `output/CNN_08.29_20-34-27/85.pth`
- Training: gamma=0.8, 100 epochs

## Figure Status

| Figure | Status | Source |
|--------|--------|--------|
| Fig.5 Training curves | Generated | `figures/training_curves.png` |
| Fig.6 Rate vs K | Generated | `figures/Fig6_rate_vs_K.png` |
| Fig.7 Rate vs alpha_N | Not generated | Requires separate gamma-trained models |
| Fig.8 Rate vs Rmin | Not generated | Requires separate Rmin-trained models |
| Fig.9 Rate vs P | Generated | `figures/Fig9_rate_vs_P.png` |
| Table I Accuracy vs SNR | Generated | `figures/tables.txt` |
| Table II Parameters | Generated | `figures/tables.txt` |

## Known Issues

1. **Classification accuracy ~96% vs paper ~99%**: Gap cannot be explained by any single code difference. Verified: random K (+0.09%), data split (+0.21%), model architecture (identical), loss (identical). Residual ~3% gap likely from Data_user.mat generation differences or paper reporting methodology.

2. **Fig.7/8 not generated**: These require training models with different gamma/Rmin values respectively. Current eval scripts only evaluate the base model (gamma=0.4). Post-processing a single model to produce Fig.7/8 curves is not valid — each point requires a separately trained model.

3. **SMR_loss indentation bug**: `SINR_k` and `rate_k` are computed inside the inner `for j` loop (should be outside). This bug exists in both author's original code and current code, so it does not affect relative comparisons.
