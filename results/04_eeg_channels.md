# EEG Channel Comparison

**Finding:** No single EEG channel is clearly best for From-Scratch or Next-Token. BIOT's EEG2 is clearly weakest.

- Every single channel scores 3 to 6 points of macro F1 below its model's full 3-channel montage.
- BIOT loses with EEG2 on 5 of 5 folds against both EEG1 and EEG3.
- From-Scratch and Next-Token channel differences are smaller than fold-to-fold noise.

**Setup:** Full MESA cohort, 2056 subjects, fold5_v1. Single-channel patience 10 for all nine runs.

**Legend:** 🟦 best single channel in column, within model, bold value. 🟩 all-3-channel row.

Derivations: EEG1 = Fz(+)-Cz(-), EEG2 = Cz(+)-Oz(-), EEG3 = C4(+)-M1(-).

## From-Scratch

| Channel | Patience | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|---|
| EEG1 | 10 | 0.6835 ± 0.0120 | **0.8010** 🟦 | 0.9176 | 0.3200 | **0.7826** 🟦 | 0.6291 | **0.7681** 🟦 |
| EEG2 | 10 | 0.6843 ± 0.0100 | 0.7978 | **0.9226** 🟦 | **0.3526** 🟦 | 0.7694 | 0.6231 | 0.7536 |
| EEG3 | 10 | **0.6886 ± 0.0121** 🟦 | 0.7994 | 0.9169 | 0.3394 | 0.7746 | **0.6456** 🟦 | 0.7663 |
| 3-channel, patience 10 | 10 | 0.7306 ± 0.0102 🟩 | 0.8309 | 0.9431 | 0.4270 | 0.8036 | 0.6583 | 0.8211 |
| 3-channel, patience 50 (benchmark) | 50 | 0.7451 ± 0.0052 🟩 | 0.8403 | 0.9468 | 0.4661 | 0.8182 | 0.6569 | 0.8373 |

## Next-Token

| Channel | Patience | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|---|
| EEG1 | 10 | **0.7015 ± 0.0063** 🟦 | **0.8084** 🟦 | 0.9218 | 0.3831 | **0.7901** 🟦 | 0.6353 | 0.7769 |
| EEG2 | 10 | 0.6966 ± 0.0098 | 0.8044 | **0.9274** 🟦 | **0.3912** 🟦 | 0.7787 | 0.6323 | 0.7534 |
| EEG3 | 10 | 0.6964 ± 0.0091 | 0.8081 | 0.9213 | 0.3412 | 0.7857 | **0.6501** 🟦 | **0.7835** 🟦 |
| 3-channel, patience 10 | 10 | 0.7346 ± 0.0053 🟩 | 0.8355 | 0.9425 | 0.4220 | 0.8148 | 0.6708 | 0.8229 |
| 3-channel, patience 50 (benchmark) | 50 | 0.7394 ± 0.0051 🟩 | 0.8373 | 0.9444 | 0.4443 | 0.8158 | 0.6684 | 0.8241 |

## BIOT

| Channel | Patience | Macro F1 | Accuracy | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|---|---|
| EEG1 | 10 | 0.6976 ± 0.0052 | 0.7795 | 0.9070 | 0.4824 | **0.7567 🟦** | 0.6244 | **0.7175** 🟦 |
| EEG2 | 10 | 0.6887 ± 0.0088 | 0.7765 | **0.9149** 🟦 | **0.4829** 🟦 | 0.7437 | 0.6174 | 0.6845 |
| EEG3 | 10 | **0.6998 ± 0.0088** 🟦 | **0.7815** 🟦 | 0.9181 | 0.4800 | 0.7462 | **0.6396** 🟦 | 0.7152 |
| 3-channel, patience 20 (benchmark) | 20 | 0.7541 ± 0.0047 🟩 | 0.8274 | 0.9455 | 0.5363 | 0.8023 | 0.6798 | 0.8068 |

## Paired per-fold win counts

| Model | Pair | Wins | Mean diff | Per-fold diff |
|---|---|---|---|---|
| From-Scratch | EEG1 vs EEG2 | EEG1 3/5, EEG2 2/5 | -0.0008 | +0.0101, -0.0106, -0.0147, +0.0062, +0.0051 |
| From-Scratch | EEG1 vs EEG3 | EEG1 2/5, EEG3 3/5 | -0.0051 | -0.0067, -0.0028, -0.0337, +0.0011, +0.0165 |
| From-Scratch | EEG2 vs EEG3 | EEG2 2/5, EEG3 3/5 | -0.0043 | -0.0167, +0.0078, -0.0190, -0.0051, +0.0115 |
| Next-Token | EEG1 vs EEG2 | EEG1 3/5, EEG2 2/5 | +0.0049 | -0.0003, +0.0101, +0.0044, -0.0090, +0.0192 |
| Next-Token | EEG1 vs EEG3 | EEG1 3/5, EEG3 2/5 | +0.0051 | -0.0020, +0.0085, +0.0073, -0.0059, +0.0175 |
| Next-Token | EEG2 vs EEG3 | EEG2 2/5, EEG3 3/5 | +0.0002 | -0.0017, -0.0015, +0.0030, +0.0031, -0.0017 |
| BIOT | EEG1 vs EEG2 | **EEG1 5/5** 🟦, EEG2 0/5 | +0.0089 | +0.0082, +0.0031, +0.0173, +0.0051, +0.0110 |
| BIOT | EEG1 vs EEG3 | EEG1 1/5, EEG3 4/5 | -0.0022 | -0.0015, -0.0090, -0.0028, -0.0045, +0.0067 |
| BIOT | EEG2 vs EEG3 | EEG2 0/5, **EEG3 5/5** 🟦 | -0.0112 | -0.0097, -0.0121, -0.0201, -0.0096, -0.0043 |

Only BIOT's EEG2-worst result clears the bar of winning most folds with a gap bigger than the fold-to-fold std.

### Per-fold macro F1

**From-Scratch**

| Channel | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| EEG1 | 0.6899 | 0.6919 | 0.6610 | 0.6934 | 0.6812 |
| EEG2 | 0.6798 | 0.7025 | 0.6757 | 0.6872 | 0.6761 |
| EEG3 | 0.6966 | 0.6947 | 0.6947 | 0.6923 | 0.6646 |
| 3-channel, patience 10 | 0.7139 | 0.7446 | 0.7324 | 0.7357 | 0.7265 |

**Next-Token**

| Channel | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| EEG1 | 0.6960 | 0.7136 | 0.6978 | 0.7002 | 0.6996 |
| EEG2 | 0.6963 | 0.7036 | 0.6935 | 0.7092 | 0.6804 |
| EEG3 | 0.6981 | 0.7051 | 0.6905 | 0.7060 | 0.6821 |
| 3-channel, patience 10 | 0.7270 | 0.7381 | 0.7373 | 0.7408 | 0.7297 |

**BIOT**

| Channel | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| EEG1 | 0.6922 | 0.7074 | 0.6952 | 0.6951 | 0.6982 |
| EEG2 | 0.6840 | 0.7043 | 0.6779 | 0.6900 | 0.6872 |
| EEG3 | 0.6937 | 0.7164 | 0.6980 | 0.6996 | 0.6915 |
