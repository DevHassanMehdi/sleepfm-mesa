# Patience-20 comparison: MESA vs. Moody (canonical copy)

This is the canonical copy of the MESA-vs-Moody patience-20 diff table.
The Moody retraining notes (owner-only,
`/scratch/project_2019517/moody2026_pilot/MATCHED_PATIENCE20_RESULTS.md`)
keep their own copy (since it's a comparison, not a pure-MESA or
pure-Moody fact) plus the Moody-only numbers, and link back here for
the MESA half.

## From-Scratch, patience 20

| Condition | MESA | Moody | MESA − Moody | Per-fold diff |
|---|---|---|---|---|
| EEG_ONLY | 0.7428±0.0045 | 0.7131±0.0088 | +0.0297 | [0.0251, 0.0367, 0.0233, 0.0500, 0.0133] |
| EEG3_ONLY | 0.6998±0.0061 | 0.6955±0.0116 | +0.0043 | [-0.0094, 0.0219, -0.0018, 0.0212, -0.0102] |
| EEG_ECG | 0.7395±0.0044 | 0.7104±0.0060 | +0.0291 | [0.0254, 0.0349, 0.0194, 0.0441, 0.0219] |
| ECG_ONLY | 0.5431±0.0126 | 0.5062±0.0132 | +0.0369 | [0.0466, 0.0333, -0.0047, 0.0716, 0.0376] |

## Next-Token, patience 20

| Condition | MESA | Moody | MESA − Moody | Per-fold diff |
|---|---|---|---|---|
| EEG_ONLY | 0.7350±0.0051 | 0.7033±0.0080 | +0.0317 | [0.0401, 0.0310, 0.0221, 0.0460, 0.0197] |
| EEG3_ONLY | 0.7138±0.0073 | 0.6983±0.0080 | +0.0155 | [0.0048, 0.0287, 0.0137, 0.0262, 0.0043] |
| EEG_ECG | 0.7322±0.0024 | 0.7020±0.0076 | +0.0302 | [0.0381, 0.0324, 0.0217, 0.0378, 0.0212] |
| ECG_ONLY | 0.4190±0.0049 | 0.3742±0.0107 | +0.0448 | [0.0566, 0.0536, 0.0255, 0.0545, 0.0338] |

## BIOT, patience 20 (MESA EEG3_ONLY is patience 10 — no p20 run exists for MESA)

| Condition | MESA | Moody | MESA − Moody | Per-fold diff |
|---|---|---|---|---|
| EEG_ONLY | 0.7541±0.0047 | 0.6932±0.0096 | +0.0609 | [0.0582, 0.0629, 0.0448, 0.0737, 0.0650] |
| EEG3_ONLY | 0.6998±0.0088 (p10) | 0.6693±0.0057 (p20) | n/a — patience mismatch | — |
| EEG_ECG | 0.7479±0.0044 | 0.6832±0.0046 | +0.0647 | [0.0707, 0.0688, 0.0538, 0.0733, 0.0568] |
| ECG_ONLY | 0.3235±0.0101 | 0.2591±0.0073 | +0.0644 | [0.0570, 0.0641, 0.0663, 0.0606, 0.0738] |

## Pattern

MESA beats Moody for every condition in every model at matched
patience 20 — a consistent cohort-size effect (2056 vs. 1135
subjects), never reversed. Smallest for EEG3_ONLY (+0.004 to +0.016,
the exact-electrode-match condition), largest for ECG_ONLY/BIOT
generally (+0.04 to +0.065).

**EEG − ECG gap, per model, per dataset** (EEG_ONLY minus ECG_ONLY):

| Model | MESA | Moody |
|---|---|---|
| From-Scratch | 0.1997 | 0.2069 |
| Next-Token | 0.3160 | 0.3291 |
| BIOT | 0.4306 | 0.4341 |
