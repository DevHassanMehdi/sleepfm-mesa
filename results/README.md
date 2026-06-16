# SleepFM MESA Sleep Staging Results

349 subjects, 10-fold CV, 5-class staging (Wake/N1/N2/N3/REM)

| Modalities | Mean Macro F1 | Std | Accuracy |
|---|---|---|---|
| BAS (EEG baseline) | 0.2122 | 0.0113 | 0.2411 |
| BAS + EKG | 0.2155 | 0.0085 | 0.2531 |
| BAS + EKG + RESP | 0.2209 | 0.0132 | 0.2578 |
| BAS + EKG + RESP + EMG | 0.2193 | 0.0074 | 0.2558 |

## Reference Points
- SleepFM paper (full multimodal, multiple cohorts): macro F1 0.70-0.78
- Hassan thesis (17 wearable features, LSTM, MESA, 4-class): macro F1 0.435
- Hassan thesis (full PSG features, XGBoost, MESA, 4-class): macro F1 0.881

## Notes
- Results use 5-class staging (Wake/N1/N2/N3/REM)
- Thesis used 4-class (AWAKE/LIGHT/DEEP/REM) — N1+N2 merged
- MESA is part of SleepFM pretraining data — data leakage limitation
- REM consistently hardest class (F1~0.09) without dedicated wearable signals
