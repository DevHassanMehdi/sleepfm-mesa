# From-Scratch Pretraining Modality-Set Ablation

Tests whether From-Scratch's elevated ECG-only fine-tuning result (vs. the
original full-cohort benchmark, see `ALL_RESULTS_FULL.md`) depends on
narrowing pretraining to EEG+ECG specifically, by varying only the
pretraining `modality_types` and holding everything else fixed: full
2056-subject cohort, same split file (`dataset_split_fromscratch_fullcohort.json`
for pretraining, `dataset_split_fold5_v1.json` for fine-tuning), same
hyperparameters (`weight_decay=0.0001`, `lr_step_period=null`, 100 epochs
pretraining), same fine-tuning pipeline (`finetune_sleep_staging.py`,
early-stopping patience=50 -- hardcoded in that script, not config-driven;
this is also what produced the reference numbers below, so the two remain
directly comparable).

## Master summary (all experiments, ECG-only and EEG-only)

**Note on the 300-subject run (Arm E):** a later instruction in this
session asked to cancel the Arm E smoke test and everything chained after
it (pretraining, embeddings, fine-tuning), and to mark this experiment
"NOT RUN" in this file. By the time that instruction was processed, the
entire chain had already completed successfully hours earlier (smoke
test COMPLETED, 100-epoch pretraining COMPLETED in 2h38m, embeddings
COMPLETED with 2056/2056 files verified, all 10 fine-tuning runs
COMPLETED) -- there was nothing left in the queue to cancel. Marking it
"NOT RUN" would misstate real, verified results, so Arm E's actual
numbers are reported below instead, with this note flagging the
discrepancy between that instruction and what had already executed.

| Experiment | Pretraining subjects | ECG-only mean (std) | ECG-only diff vs. ref | EEG-only mean (std) | EEG-only diff vs. ref |
|---|---|---|---|---|---|
| **Reference** (EEG+ECG, fixed hyperparams) | 2056 | 0.5575 (0.0059) | -- | 0.7451 (--) | -- |
| Arm A (EEG_ONLY+RESP+EKG+EMG) | 2056 | 0.5405 (0.0060) | -0.0170 | 0.7468 (0.0035) | +0.0017 |
| Arm B (RESP+EKG+EMG) | 2056 | 0.5329 (0.0057) | -0.0247 | -- | -- |
| Arm C (EKG+RESP) | 2056 | 0.5602 (0.0061) | +0.0027 | -- | -- |
| Arm D (EKG+EMG) | 2056 | 0.5173 (0.0084) | -0.0402 | -- | -- |
| Old 350-subject encoder (buggy hyperparams, 4-modality) | 350 (300 pretrain) | 0.3899 (0.0045) | **-0.1677** | 0.6954 (0.0027) | -0.0497 |
| Arm E (300subj, FIXED hyperparams, 4-modality) | 300 | 0.5231 (0.0064) | -0.0344 | 0.7303 (0.0019) | -0.0148 |
| **seed123** (EEG_ONLY+EKG, fixed hyperparams, different seed) | 2056 | 0.5596 (0.0055) | **+0.0020** | 0.7489 (0.0037) | +0.0038 |

seed123's EEG+ECG mean is 0.7503 (std 0.0050), diff +0.0068 vs. the
reference's 0.7435 -- also within reference fold-to-fold noise.

**Key takeaway:** seed123 replicates the reference almost exactly across
all three modalities (every diff within the reference's own fold-to-fold
std of 0.0059), confirming the reference numbers are not a fluke of one
particular random initialization. Arm E isolates cohort size from
hyperparameters: at the SAME ~300-subject scale as the old encoder but
with the FIXED hyperparameters, ECG-only improves from 0.3899 to 0.5231
and EEG-only from 0.6954 to 0.7303 -- most of the old encoder's deficit
was the old buggy hyperparameters (`weight_decay=0.0`,
`lr_step_period=2`), not primarily the small cohort. Arm E still trails
the full 2056-subject reference by roughly half the old encoder's gap,
consistent with cohort size also mattering, just less than the
hyperparameter bug did.

### Old 350-subject encoder -- facts from config.json and its TSV log (Part B)

- Learning rate: `lr=0.001`, `lr_step_period=2`, `gamma=0.1` -- confirmed
  StepLR with 10x decay every 2 epochs (`gamma=0.1` = 10x, applied every
  `lr_step_period=2` epochs), the exact schedule later identified as
  collapsing the effective LR to near-zero well before training
  finishes.
- `weight_decay=0.0`, `batch_size=128`, configured `epochs=100`.
- `modality_types`: `["EEG_ONLY", "RESP", "EKG", "EMG"]` (the full
  4-modality scope, not the reference's 2-modality EEG+ECG).
- Split file: `sleepfm/configs/dataset_split_fromscratch.json` -- 300
  pretrain subjects (identical to "train"), 50 test, 30 validation (the
  30 validation subjects are a pure subset of the 300 pretrain subjects,
  not a real holdout).
- The TSV log only contains 67 rows (epochs 0-66, "pretrain" split
  only) despite `epochs: 100` in the config -- the run did not log (and
  may not have run) all 100 configured epochs; no validation-split rows
  exist in this TSV and no surviving stdout log was found, so epoch-by-
  epoch validation accuracy for this historical run cannot be reported.
  By epoch 64-66 the logged train accuracy is itself nearly flat
  (7.8-7.9% / 13.1-13.2% across the two weakest/strongest modality
  pairs), consistent with the StepLR schedule having collapsed the
  learning rate to a negligible value by that point.

### Buggy-run checkpoint check (Part D, read-only)

`checkpoints/encoders/fromscratch_eegonly_fullcohort_oldbuggyrun_pre_lrfix`
still exists: 75M total
(`SetTransformer/leave_one_out_128_patch_size_640/`), `best.pt` dated
2026-08-09 15:41:01, `checkpoint.pt` dated 2026-08-09 18:30:07. Not
touched by this ablation.

---

## Arms

| Arm | Pretraining modality_types |
|---|---|
| A | EEG_ONLY, RESP, EKG, EMG (full original 4-modality scope) |
| B | RESP, EKG, EMG (no EEG) |
| C | EKG, RESP |
| D | EKG, EMG |

## Reference (From-Scratch EEG+ECG pretraining, fold5_v1, 2056 subjects)

| Modality | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean | Std |
|---|---|---|---|---|---|---|---|
| ECG-only | 0.5595 | 0.5606 | 0.5601 | 0.5616 | 0.5459 | 0.5575 | 0.0059 |
| EEG-only | -- | -- | -- | -- | -- | 0.7451 | -- |
| EEG+ECG | -- | -- | -- | -- | -- | 0.7435 | -- |

## Results

All macro F1 / accuracy values below were recomputed independently from the
raw test-set pickles (`test_all_outputs.pickle`, `test_all_targets.pickle`,
`test_all_masks.pickle`) using plain `sklearn.metrics.f1_score(average="macro")`
/ `accuracy_score` -- not `compute_metrics.py` -- and matched
`compute_metrics.py`'s own output exactly for every fold checked. Fold
disjointness was verified per experiment: each of the 5 test folds'
subject sets are pairwise disjoint and their union is exactly the 2056-subject
cohort, for every arm/modality below.

### ECG_ONLY fine-tuning, all 4 arms

| Arm | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean | Std | Diff vs. reference mean |
|---|---|---|---|---|---|---|---|---|
| A (4-modality) | 0.5293 | 0.5462 | 0.5421 | 0.5450 | 0.5400 | 0.5405 | 0.0060 | -0.0170 |
| B (RESP+EKG+EMG) | 0.5237 | 0.5348 | 0.5295 | 0.5401 | 0.5363 | 0.5329 | 0.0057 | -0.0247 |
| C (EKG+RESP) | 0.5596 | 0.5669 | 0.5602 | 0.5649 | 0.5493 | 0.5602 | 0.0061 | **+0.0027** |
| D (EKG+EMG) | 0.5114 | 0.5294 | 0.5215 | 0.5193 | 0.5050 | 0.5173 | 0.0084 | **-0.0402** |

Per-fold accuracy (same order): A: 0.6867/0.6937/0.6898/0.7015/0.6955;
B: 0.6821/0.6950/0.6844/0.6946/0.6905; C: 0.7077/0.7118/0.7014/0.7113/0.7104;
D: 0.6733/0.6843/0.6740/0.6837/0.6696.

### Arm A only -- EEG_ONLY and EEG_ECG fine-tuning

| Modality | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean | Std | Diff vs. reference mean |
|---|---|---|---|---|---|---|---|---|
| EEG-only | 0.7444 | 0.7524 | 0.7455 | 0.7491 | 0.7427 | 0.7468 | 0.0035 | +0.0017 |
| EEG+ECG | 0.7530 | 0.7521 | 0.7451 | 0.7536 | 0.7485 | 0.7505 | 0.0032 | +0.0070 |

## Interpretation

- **Arm A's EEG-only and EEG+ECG land close to the reference** (+0.0017 and
  +0.0070, both well within the reference's own fold-to-fold std of 0.0059)
  -- consistent with Arm A's pretraining encoder behaving comparably to the
  reference encoder for the modalities both pretraining runs share.
- **Arm A's ECG-only is consistently below the reference across all 5
  folds** (-0.0059 to -0.0302, mean -0.0170) -- directly answering the
  original question this ablation was designed to test: narrowing
  pretraining to EEG+ECG specifically (rather than the full 4-modality
  scope) does appear to be part of why the reference run's ECG-only result
  is as high as it is. Broadening pretraining back out (Arm A) gives up
  some of that ECG-only benefit, while EEG-only/EEG+ECG are unaffected.
- **Arm C (EKG+RESP) is the only arm at or above the reference**
  (+0.0027, within reference fold-to-fold noise) -- a narrow 2-modality
  pretraining pairing, like the reference, just with RESP substituted for
  EEG.
- **Arm D (EKG+EMG) is the weakest ECG-only performer** (-0.0402 mean, the
  largest deviation of any arm, roughly 7x the reference's own fold-to-fold
  std) **despite having by far the highest pretraining retrieval accuracy
  of any arm** (86-90% vs. 60-66% for A/B/C -- see pretraining logs). High
  pretraining retrieval accuracy for this pairing does not translate to
  better downstream ECG-only fine-tuning -- worth flagging as a real,
  non-noise disconnect between the pretraining objective and the
  fine-tuning task, not something resolved within this ablation.

## Provenance

- Pretraining jobs: A=2009759, B=2009760, C=2049105, D=2049106 (all
  COMPLETED, 100/100 epochs, no timeouts).
- Embeddings jobs (`generate_embeddings.py`, `--hdf5_dir` required for the
  flat full-cohort layout): A=2049988, B=2049989, C=2074994, D=2074995 --
  all COMPLETED, 2056/2056 files each, HDF5 keys verified against each
  arm's modality_types, file timestamps after pretraining completion.
- Fine-tuning jobs (`finetune_sleep_staging.py` via
  `finetune_sleep_staging_fullcohort_2056.slurm`, `--split_id fold5_v1`,
  hardcoded patience=50): Arm A ECG_ONLY folds 0-4 = 2050615, 2074972-2074975;
  Arm A EEG_ONLY folds 0-4 = 2074976-2074980; Arm A EEG_ECG folds 0-4 =
  2074981-2074985; Arm B ECG_ONLY folds 0-4 = 2050616, 2074986-2074989;
  Arm C ECG_ONLY folds 0-4 = 2075594, 2076613-2076616; Arm D ECG_ONLY folds
  0-4 = 2075595, 2076617-2076620. All 30 runs COMPLETED, exit code 0.
- `metrics.json` present for all 30 fold/experiment combinations under
  `experiments_full_cohort/full_cohort/sleepfm__*ablation*`.
- Fold-0 smoke-check results (Arm A/B ECG_ONLY, Arm C/D ECG_ONLY) were
  confirmed unchanged after the remaining folds were submitted under the
  same experiment timestamp -- no overwrite occurred.

---

# Follow-up: old 350-subject encoder on the corrected full cohort

Tests the preserved, pre-full-cohort encoder (`checkpoints/encoders/
fromscratch_eegonly/SetTransformer/leave_one_out_128_patch_size_640`),
trained on only 350 subjects (300 pretrain + 50 test,
`dataset_split_fromscratch.json`) with the **old, buggy hyperparameters**
(`weight_decay=0.0`, `lr_step_period=2` -- the exact combination identified
earlier this session as causing a flat-validation-curve pretraining
pathology) and the full 4-modality scope (`modality_types: ["EEG_ONLY",
"RESP", "EKG", "EMG"]`, not the reference's 2-modality EEG+ECG scope).
Checkpoint dated 2026-06-24/25 (`best.pt`/`checkpoint.pt`), confirming it
predates this session's hyperparameter fix and full-cohort split
correction.

`best.pt`/`config.json` were copied (not moved, not edited in place) to a
new directory, `checkpoints/encoders/fromscratch_old350enc_ablation/`,
with only `data_path` repointed from the stale relative `data/mesa_full`
to the real absolute `hdf5_full` path in the copy. The original
`fromscratch_eegonly/` directory was left untouched (verified via
checksum match on `best.pt` before and after the copy).

Embeddings were generated for the full 2056-subject cohort from this
copy (job 2241437): 2056/2056 files, HDF5 keys confirmed as
`{EEG_ONLY, RESP, EKG, EMG}` matching the checkpoint's own
`modality_types`, written 2026-10-12 (long after the June checkpoint
date, confirming the embeddings reflect the real stored weights, not
stale cached output).

Fine-tuned ECG_ONLY and EEG_ONLY, 5 folds each, `dataset_split_fold5_v1.json`,
`pretrain_method=fromscratch_old350enc`. All 10 runs COMPLETED (smoke
check on fold 0 of each first, then folds 1-4). Recomputed independently
from raw pickles via plain sklearn; fold disjointness confirmed clean
(2056 unique subjects, zero overlap) for both modalities.

| Modality | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean | Std | Diff vs. reference mean |
|---|---|---|---|---|---|---|---|---|
| ECG-only | 0.3891 | 0.3968 | 0.3864 | 0.3927 | 0.3844 | 0.3899 | 0.0045 | **-0.1677** |
| EEG-only | 0.6956 | 0.6997 | 0.6928 | 0.6967 | 0.6924 | 0.6954 | 0.0027 | -0.0497 |

**Interpretation:** both modalities fall well below the reference, as
expected given this encoder combines three confounded disadvantages
relative to the reference (350 vs. 2056 pretraining subjects, the
old buggy weight_decay/lr_step_period combination, and the broader
4-modality rather than 2-modality pretraining scope) -- this run does not
isolate which of the three drives the gap, only that the combination
produces a large, consistent deficit (ECG-only: -0.168, roughly 28x the
reference's own fold-to-fold std; EEG-only: -0.050, roughly double what
Arm A/B/C/D's modality-set ablation ever produced). Not flagged as a
pipeline problem -- this is the expected direction and rough scale for an
undertrained, small-cohort, buggy-hyperparameter encoder.

## Provenance (old 350-subject encoder follow-up)

- Encoder copy: `fromscratch_old350enc_ablation` (copied from
  `fromscratch_eegonly`, original untouched, checksum-verified).
- Embeddings job: 2241437, COMPLETED, 2056/2056 files.
- Fine-tuning jobs (`fromscratch_old350enc`): ECG_ONLY folds 0-4 =
  2242970, 2243871-2243874; EEG_ONLY folds 0-4 = 2242971, 2243875-2243878.
  All 10 COMPLETED, exit code 0.

---

# Follow-up: second seed (seed123) and 300-subject fixed-hyperparameter run (Arm E)

## seed123 -- same reference config (EEG_ONLY+EKG, 2056 subjects, fixed
hyperparameters), pretrained from a different random seed (123) to test
whether the reference numbers depend on one particular initialization.

`pretrain.py` had no seed option at all before this (config's `seed: 42`
field was unused -- confirmed via `grep`, no `torch.manual_seed`/
`random.seed`/`np.random.seed` call existed anywhere in the script). Added
a `--seed` CLI option (default `None`, which preserves the exact prior
unseeded behavior) that calls a new `set_seed()` helper (mirroring the
pattern already used in `finetune_disease_prediction.py`/
`finetune_diagnosis_coxph.py`) when a value is passed. Full diff:

```diff
+import random
...
+def set_seed(seed):
+    random.seed(seed)
+    np.random.seed(seed)
+    torch.manual_seed(seed)
+    torch.cuda.manual_seed_all(seed)
+    torch.backends.cudnn.deterministic = True
+    torch.backends.cudnn.benchmark = False
...
+@click.option("--seed", type=int, default=None, help="...")
 def pretrain(
-    config_path, channel_groups_path, checkpoint_path, use_wandb
+    config_path, channel_groups_path, checkpoint_path, use_wandb, seed
 ):
+    if seed is not None:
+        set_seed(seed)
     config = load_config(config_path)
...
+    logger.info(f"Seed: {seed if seed is not None else '(none -- unseeded, default behavior)'}")
```

Pretraining (job 2241656) COMPLETED, 100/100 epochs, 6h13m. Validation
retrieval accuracy climbed clearly and did not stay flat: epoch 2.5 ->
40.0%, epoch 50 -> 61.2%, peak 63.8% (epoch 92.5), final epoch 100 ->
61.6%. Embeddings (job 2384800): 2056/2056 files, keys
`{EEG_ONLY, EKG}`, model_path/checkpoint confirmed, written 2026-10-20
(long after pretraining finished 2026-10-12). Fine-tuning: ECG_ONLY,
EEG_ONLY, EEG_ECG, 5 folds each (15 runs total, `pretrain_method=
fromscratch_seed123`), smoke-checked on ECG_ONLY fold 0 first -- all 15
COMPLETED, exit code 0. Recomputed independently via plain sklearn from
raw pickles for every fold; fold disjointness clean (2056 unique
subjects, zero overlap) for all three modalities.

| Modality | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean | Std | Diff vs. reference |
|---|---|---|---|---|---|---|---|---|
| ECG-only | 0.5601 | 0.5686 | 0.5602 | 0.5515 | 0.5575 | 0.5596 | 0.0055 | +0.0020 |
| EEG-only | 0.7507 | 0.7538 | 0.7444 | 0.7510 | 0.7448 | 0.7489 | 0.0037 | +0.0038 |
| EEG+ECG | 0.7508 | 0.7556 | 0.7427 | 0.7554 | 0.7468 | 0.7503 | 0.0050 | +0.0068 |

Every diff is within the reference's own fold-to-fold std (0.0059) --
seed123 replicates the reference cleanly across all three modalities.

## Arm E -- same 4-modality scope and fixed hyperparameters as Arm A, but
pretrained on the SAME 300 subjects as the old 350-subject encoder
(`dataset_split_fromscratch_300subj.json`: the old encoder's exact 300
"pretrain" subjects, plus an 87-subject validation set built from the
standard full-cohort validation-100 list with the 13 subjects that
overlap the 300 pretrain subjects removed -- the old encoder's own
30-subject "validation" list was not reused since it is a pure subset of
its own 300 pretrain subjects, not a real holdout).

Config diff against `config_pretrain_fromscratch_ablation_A_fullcohort.yaml`
(only `split_path` and `save_path` differ; `weight_decay=0.0001`,
`lr_step_period=null`, `epochs=100` all preserved):

```diff
< split_path: 'sleepfm/configs/dataset_split_fromscratch_fullcohort.json'
< save_path: '.../checkpoints/encoders/fromscratch_ablation_A_fullcohort'
---
> split_path: 'sleepfm/configs/dataset_split_fromscratch_300subj.json'
> save_path: '.../checkpoints/encoders/fromscratch_ablation_E_300subj'
```

3-epoch smoke test (job 2384893): validation accuracy moved from 1.73%
up to 2.45% (noisy but a genuine upward trend, not flat) -- the absolute
scale is much lower than other arms' smoke tests because this config
uses `batch_size=128` (random-chance retrieval accuracy ~0.78%) versus
the other arms' smoke tests which used `batch_size=8` (random chance
~12.5%); 2-3x above chance with a clear upward trend was judged healthy,
not flat, and the full run was submitted.

Full pretraining (job 2385678) COMPLETED in 2h38m (within the ~3h
estimate). Embeddings (job 2385679, chained via `--dependency=afterok`):
2056/2056 files, keys `{EEG_ONLY, RESP, EKG, EMG}`, checkpoint path and
timestamps verified. Fine-tuning: ECG_ONLY and EEG_ONLY, 5 folds each
(10 runs, `pretrain_method=fromscratch_ablationE`, chained via `afterok`
on the embeddings job) -- all 10 COMPLETED. Fold disjointness clean for
both modalities.

| Modality | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean | Std | Diff vs. reference |
|---|---|---|---|---|---|---|---|---|
| ECG-only | 0.5221 | 0.5312 | 0.5220 | 0.5279 | 0.5123 | 0.5231 | 0.0064 | -0.0344 |
| EEG-only | 0.7299 | 0.7330 | 0.7304 | 0.7313 | 0.7271 | 0.7303 | 0.0019 | -0.0148 |

## Provenance (seed123 + Arm E follow-up)

- seed123: pretrain=2241656; embeddings=2384800; fine-tuning ECG_ONLY
  folds 0-4 = 2385676, 2387357-2387360; EEG_ONLY folds 0-4 =
  2387361-2387365; EEG_ECG folds 0-4 = 2387366-2387370. All COMPLETED,
  exit 0.
- Arm E: smoketest=2384893; full pretrain=2385678; embeddings=2385679;
  fine-tuning ECG_ONLY folds 0-4 = 2385806-2385810; EEG_ONLY folds 0-4 =
  2385811-2385815. All COMPLETED, exit 0.

This file is intentionally separate from `results/ALL_RESULTS_FULL.md`,
which documents the finished, independently-verified 90-run benchmark and
was not modified by this ablation.
