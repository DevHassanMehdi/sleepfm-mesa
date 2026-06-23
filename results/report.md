# SleepFM on MESA: Sleep Staging Report

## Summary

We built a sleep staging pipeline on the MESA dataset using SleepFM. We ran it two ways. First, with the published SleepFM encoder, which gives results in the range the paper reports (around 0.72 to 0.73 macro F1). Second, we trained an encoder from scratch on MESA only, to remove the data leakage that comes from the published encoder having seen MESA during its own pretraining. The from-scratch model reaches around 0.65 macro F1 on a fully held-out test set.

This report explains what we did, the exact configuration so others can reproduce it, and the results for each setup.

## Background

SleepFM is a multimodal sleep foundation model. It encodes raw polysomnography (PSG) signals into embeddings, then trains lightweight heads on those embeddings for downstream tasks. We focus on 5-class sleep staging: Wake, N1, N2, N3, and REM.

One important caveat. The published SleepFM encoder was pretrained on several cohorts, and MESA was one of them. So when we use the published encoder on MESA, the encoder has already seen our data. The test subjects are held out from the staging head, but not from the encoder. This is a form of leakage. To address it, we also trained an encoder from scratch on MESA training subjects only, with a held-out test set the encoder never saw.

## Data

- Dataset: MESA polysomnography, 350 subjects(mesa-sleep-0001---mesa-sleep-1215), from the NSRR release.
- Signals used, grouped into four modalities:
  - BAS: 3 EEG channels (EEG1, EEG2, EEG3) and 2 EOG channels (EOG-L, EOG-R)
  - EKG: 1 channel
  - RESP: 6 channels (Abdo, HR, Snore, SpO2, Therm, Thor)
  - EMG: 3 channels (EMG, Leg, Pleth)
- MESA does not have skin-temperature or accelerometer channels.
- Labels: sleep stage annotations from the NSRR XML files, expanded to one label per 30-second epoch.

### Label note (a bug we fixed)

The MESA annotation files store sleep stages as variable-length events. For example, one row can cover 90 minutes of Wake. Our pipeline originally read each row as a single 30-second epoch, which misaligned every label after the first with the signal. This capped staging performance at chance level. We fixed it by expanding each stage event into one row per 30-second epoch based on its duration. After this fix, performance jumped into the expected range.

## Preprocessing

This matches the SleepFM paper.

- Resample all signals to 128 Hz.
- Apply a 4th-order Butterworth low-pass anti-alias filter before resampling.
- Z-score normalize each channel, per recording.
- Segment signals into 5-second windows (640 samples at 128 Hz). These windows are the model's input tokens.

## Model and pipeline configuration

### Encoder (SetTransformer)

The encoder turns raw signals into embeddings. It uses 1D convolutional layers to tokenize each 5-second window, channel-agnostic attention pooling to combine channels within a modality, and a temporal transformer over a 5-minute context.

- Model type: SetTransformer
- embed_dim: 128
- num_heads: 8
- num_layers: 6
- pooling_head: 8
- patch_size: 640 (5 seconds at 128 Hz)
- in_channels: 1
- sampling_duration: 5, sampling_freq: 128
- Modality groups: BAS, RESP, EKG, EMG

The encoder produces a 128-dimensional embedding per modality, at 5-second resolution.

### Staging head (SleepEventLSTMClassifier)

The head is trained on the frozen embeddings.

- Model type: SleepEventLSTMClassifier
- 2-layer bidirectional LSTM
- embed_dim: 128
- num_heads: 4
- num_layers: 2
- pooling_head: 4
- dropout: 0.3
- num_classes: 5
- context: -1 (full night per sample)
- max_seq_length: 8196

### Staging training

- Learning rate: 1e-4
- Max epochs: 500
- Early stopping patience: 50 (stops if validation macro F1 does not improve for 50 epochs)
- Loss: masked cross-entropy with class weights
- Optimizer: Adam

### From-scratch pretraining (for the no-leakage experiment)

- Objective: leave-one-out multimodal contrastive learning
- Pretraining data: 270 MESA training subjects only
- Epochs: 100 (the contrastive objective converged and plateaued early, around epoch 4 to 5, with contrastive accuracy near 7 percent; this is expected for a small dataset)
- Learning rate: 1e-3
- Batch size: 128

## Experimental setup

### Published encoder

- Encoder: published SleepFM checkpoint (frozen)
- Evaluation: 10-fold cross validation across all 350 subjects
- For each fold, the staging head is trained on the training subjects and evaluated on the held-out test fold

### From-scratch encoder

- Encoder: trained by us on MESA training subjects only
- Split: 270 train, 30 validation, 50 held-out test
- The 50 test subjects are held out from both pretraining and the staging head
- Evaluation: single split, on the 50 held-out test subjects (49 have labels)

## Results

### Published encoder (10-fold CV, all 350 subjects)

| Modalities | Macro F1 | Accuracy |
| --- | --- | --- |
| BAS (EEG/EOG) | 0.7205 | 0.8216 |
| BAS + EKG | 0.7237 | 0.8232 |
| BAS + EKG + RESP | 0.7269 | 0.8253 |
| BAS + EKG + RESP + EMG | 0.7305 | 0.8259 |

Per-class F1, full multimodal: Wake 0.94, N1 0.43, N2 0.80, N3 0.65, REM 0.84.

Observations:
- Each added modality improves macro F1 by a small amount, about 0.01 in total.
- Fold variance is low, around 0.012 to 0.015.
- Results are within the SleepFM paper range of 0.70 to 0.78.
- N1 is the weakest stage, which is expected since it is the hardest stage to score and human agreement on it is low.

### From-scratch encoder (single split, 50 held-out test subjects, no leakage)

| Modalities | Macro F1 | Accuracy |
| --- | --- | --- |
| BAS (EEG/EOG) | 0.6509 | 0.7702 |
| BAS + EKG | 0.6310 | 0.7610 |
| BAS + EKG + RESP | 0.6496 | 0.7755 |
| BAS + EKG + RESP + EMG | 0.6601 | 0.7802 |

Per-class F1, full multimodal: Wake 0.90, N1 0.27, N2 0.76, N3 0.62, REM 0.75.

Observations:
- Removing leakage and pretraining on only 270 subjects costs about 0.07 to 0.09 macro F1.
- The from-scratch model still reaches the mid 0.60s, which shows the architecture, the staging head, and the data carry real signal even without large-scale pretraining.
- The modality ablation is noisier here and not cleanly monotonic. The full multimodal setup is still the best at 0.66. This is expected with a single split rather than 10-fold.
- N1 is again the weakest stage.

## Comparison and interpretation

| Setup | Leakage | Protocol | Macro F1 |
| --- | --- | --- | --- |
| Published encoder | Yes (encoder saw MESA) | 10-fold CV test | 0.72 to 0.73 |
| From-scratch encoder | No | held-out 50 test | 0.63 to 0.66 |

The gap between the two is about 0.07 to 0.09 macro F1. This is the cost of removing leakage and pretraining on a small dataset. The fact that the from-scratch model still reaches the mid 0.60s suggests that much of the staging performance comes from the architecture, the staging head, and the data itself, not only from large-scale pretraining or from MESA leakage.


## How to reproduce

Key files in the repository:

- Label generation: `scripts/generate_labels.py` (expands stage events into per-30-second-epoch labels)
- Published-encoder ablation configs: `sleepfm/configs/config_ft_BAS.yaml`, `config_ft_BAS_EKG.yaml`, `config_ft_BAS_EKG_RESP.yaml`, `config_ft_BAS_EKG_RESP_EMG.yaml`
- Published-encoder split: `data/mesa/dataset_split_10fold.json`
- Published-encoder ablation launcher: `scripts/submit_experiment.sh`
- From-scratch split: `sleepfm/configs/dataset_split_fromscratch.json` (270 train, 30 val, 50 test)
- From-scratch pretraining config: `sleepfm/configs/config_pretrain_fromscratch.yaml`
- From-scratch pretraining launcher: `scripts/pretrain_fromscratch.slurm`
- From-scratch staging launcher: `scripts/fromscratch_staging.slurm`
- Held-out test split for from-scratch eval: `sleepfm/configs/fromscratch_heldout_flat.json`

Result files:

- `results/modality_ablation_summary.txt` (published encoder)
- `results/fromscratch_heldout_summary.txt` (from-scratch, held-out test)
- `results/comparison_published_vs_fromscratch.txt` (side by side)

Environment:

- Run on the CSC Puhti supercomputer, GPU partition, one V100 per job.
- Conda environment `sleepfm_env`, Python 3.10.
- Set `export HDF5_USE_FILE_LOCKING=FALSE` before any HDF5 operation.
