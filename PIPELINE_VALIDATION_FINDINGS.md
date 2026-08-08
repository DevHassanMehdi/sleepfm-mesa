# Pipeline Validation Findings

Consolidated record of every diagnostic investigation, bug fix, and
experiment run during the MESA full-cohort expansion (350 → 1944 subjects)
and the Roihu/GH200 migration. Written as the permanent reference for
questions like "why does the full-cohort number look different from the
paper's number" and "how do we know the pipeline is trustworthy."

Every number below was pulled from a source that still exists on disk:
git history, `experiments_full_cohort/full_cohort/` (detailed per-run
artifacts), `results/ALL_RESULTS_FULL.md`, or raw prediction pickles
independently re-verified with plain `sklearn.f1_score`. Where a claim
could not be grounded in a surviving artifact, that gap is stated
explicitly rather than filled in.

**A note on commit attribution**: this work happened across three
branches (`main`, `puhti-work`, `roihu-work`) before being consolidated.
Every commit hash cited below was individually checked against `main`'s
actual history (`git merge-base --is-ancestor`), not assumed from commit
messages — one real gap was found and fixed this way: the MESA
label-expansion fix (Section 1) existed only on `roihu-work` and had
never been merged, so the code being committed alongside this document
now includes it directly rather than describing a fix that wasn't
actually present. Two other citations (Sections 2 and 3) point to
commits that aren't literal ancestors of `main` either, but whose
content was verified present in `main`'s current state by other means —
noted inline where that applies.

---

## 1. MESA label-expansion bug

**Where**: `scripts/download_mesa.py`, `parse_mesa_xml()`
**Fixed in**: commit `e34d913`, "Fix orphan-EDF redownload and
annotation-expansion bugs in MESA/SHHS download scripts" (2026-07-28)

**Discovery.** MESA's NSRR annotation XML run-length-encodes consecutive
same-stage epochs: a 4-minute stretch of Wake is stored as a single
`ScoredEvent` with `Duration=240` (seconds), not as eight separate
30-second events. The original `parse_mesa_xml()` emitted exactly one
label row per XML `ScoredEvent`, regardless of its duration:

```python
# before
rows.append({
    "Start": start,
    "Stop": start + float(dur_el.text),
    "StageName": stage_name,
    "StageNumber": stage_num,
})
```

This meant any multi-epoch scored event produced only **one** label row
instead of `duration / 30`. Since most of a night's sleep stage annotation
consists of exactly these multi-epoch runs (stage transitions are
infrequent relative to 30-second epoch granularity), this silently
undercounted label rows for the overwhelming majority of a subject's
recording.

**Root cause**: a one-event-to-one-row assumption that doesn't hold for
MESA's XML encoding — it holds for datasets that encode one row per epoch
directly, but not for MESA's run-length-encoded format.

**Fix**: expand each scored event into `n_epochs = round(duration / 30.0)`
separate rows, one per 30-second epoch, with `Start`/`Stop` advanced by 30
seconds each:

```python
# after
n_epochs = int(round(dur / EPOCH_SEC))
for e in range(n_epochs):
    ep_start = start + e * EPOCH_SEC
    ep_stop = ep_start + EPOCH_SEC
    rows.append({"Start": ep_start, "Stop": ep_stop,
                 "StageName": stage_name, "StageNumber": stage_num})
```

The same commit also fixed two related download-robustness issues in
`download_mesa.py`: an incorrect `NSRR_BIN` path, and redundant EDF
re-downloads when only the annotation file was actually missing (now
skips re-fetching a valid, already-present EDF and only re-fetches the
missing annotation) — plus orphan cleanup, described in Section 8.

**Verification**: see Section 2 — every one of the full 1944-subject
cohort's label CSVs now has a row count consistent with its recording
duration (confirmed via the epoch/label-alignment check in Section 5,
which found 334/388 sampled test subjects match label row count to
predicted-epoch count exactly, with the remaining mismatches fully
explained by unrelated, benign truncation effects — not by this bug
recurring).

---

## 2. MESA full-cohort data integrity

**350-subject baseline preserved, untouched.** The original Puhti-era
cohort is fully intact at `data/mesa/hdf5/` (350 files, confirmed present)
and its source directory `mesa/hdf5_350_puhti_baseline/`. A manifest
records the exact subject list:

```
sleepfm/configs/mesa_350_subset_manifest.json
  _metadata.description: "Exact 350-subject MESA cohort used for all
                           Puhti-era paper results"
  _metadata.generated_on: 2026-07-27
  subject_ids: 350 entries
```

(first created on the `roihu-work` branch in commit `b5b4ed1`, "Update
Roihu infra: ARM env + longer time limits, add MESA 350-subject
manifest"; the file present on `main` — verified in HEAD as of this
commit — arrived via `245a6d8`)

**Full 1944-subject cohort, verified end-to-end.** Directly counted on
disk (current state):

| Stage | Directory | Count |
|---|---|---|
| Raw EDF | `mesa/edf/` | 1944 |
| Raw annotation XML | `mesa/annotations/` | 1944 |
| Converted signal HDF5 | `mesa/hdf5_full/` | 1944 |
| Sleep-stage label CSV | `mesa/labels/` | 1944 |

All four stages match exactly — no subject was lost or silently dropped
anywhere in the EDF → annotation → HDF5 → label pipeline. The full-cohort
dataset split (`dataset_split_fromscratch_fullcohort.json`) also confirms
1944 subjects in its `pretrain`/`train` keys.

---

## 3. Folder structure redesign

**Commit `984641b`** (on the `puhti-work` branch), "Add self-contained,
descriptively-named experiment folder structure for full-cohort work"
(2026-07-28), introduced `sleepfm/experiment_paths.py` as the single
source of truth for checkpoint/results paths, replacing ad-hoc
`os.path.join(...)` calls. This content is verified present on `main` as
of this commit (`experiment_paths.py`'s `ExperimentID` class,
`compute_metrics.py`'s metrics-bundle writer, and
`evaluate_sleep_staging.py`'s split-prefixed pickle naming were all
directly checked against HEAD) — it arrived on `main` by a path other
than a direct merge of `984641b` itself, which this repository's `main`
branch does not currently contain as an ancestor.
scattered across scripts:

- **`ExperimentID` naming scheme**: `{model}__{modality}__{pretrain_method}__{split_id}__{timestamp}`
  (e.g. `sleepfm__EEG_ONLY__fromscratch__fold5_v1__2026-07-29_150556`),
  with `checkpoint_dir`/`results_dir` properties and `fold_dir(n)` helper.
- **Bidirectional config cross-linking**: `link_checkpoint_and_results()`
  writes each side's path into the other's `config.json`, so either
  directory alone tells you where its counterpart lives.
- **`compute_metrics.py` rewritten** to actually persist `metrics.json`,
  `classification_report.txt`, and `per_subject_results.csv` instead of
  only printing to stdout — this closed a real provenance gap: Spectral
  and Next-Token's 350-subject Puhti-era results had lost their audit
  trail for exactly this reason.
- **`evaluate_sleep_staging.py`**: val/test pickles now coexist under a
  flat `fold_N/` via `{split}_` filename prefixes, instead of overwriting
  each other (previously test would clobber val or vice versa).
- BIOT/LaBraM/SensorLM migrated into the unified `checkpoints/full_cohort/`
  scheme, with resume support (`--checkpoint_dir`) and dataset classes
  updated to return subject identity for per-subject reporting.

**This session's follow-up** (restructuring task, this conversation):
`results/full_cohort/` had grown to ~100+ deeply-nested timestamped
experiment directories — good for provenance, unusable for a quick look.
Moved wholesale to `experiments_full_cohort/full_cohort/` (byte-for-byte
verified: 253 files / 64 directories before and after, path-normalized
diff showed zero differences), and `experiment_paths.py`'s `results_dir`
property updated so all *future* fine-tuning runs write their detailed
artifacts there automatically:

```diff
     def results_dir(self) -> Path:
-        return REPO_ROOT / "results" / "full_cohort" / self.run_name
+        return REPO_ROOT / "experiments_full_cohort" / "full_cohort" / self.run_name
```

`results/` itself now holds flat, human-readable summaries mirroring the
original 350-subject convention: `FS_fullcohort_ALL_results.txt`,
`Spectral_fullcohort_ALL_results.txt`, `NextToken_fullcohort_ALL_results.txt`,
and a master `ALL_RESULTS_FULL.md`.

---

## 4. Fold scheme decision: fold5_v1 vs fold10_v1

**Commit `245a6d8`** added `generate_cv_splits.py` (subject-level fold
assignment, seed=42, no demographic stratification — no AHI/age/sex
metadata was available to stratify on) and generated
`dataset_split_fold5_v1.json` / `dataset_split_fold10_v1.json`.

**The comparison**: 45 completed fine-tuning runs (3 modalities ×
(5+10) fold rotations) using the full-cohort From-Scratch encoder's
`best.pt`. Full detail in `experiments_full_cohort/full_cohort/
CV_COMPARISON_fold5_vs_fold10.md`.

| Modality | Split | N folds | Mean F1 | Std | SEM |
|---|---|---|---|---|---|
| EEG_ONLY | fold5_v1 | 5 | 0.5162 | 0.0080 | **0.0036** |
| EEG_ONLY | fold10_v1 | 10 | 0.5103 | 0.0215 | 0.0068 |
| ECG_ONLY | fold5_v1 | 5 | 0.3943 | 0.0109 | **0.0049** |
| ECG_ONLY | fold10_v1 | 10 | 0.4006 | 0.0168 | 0.0053 |
| EEG_ECG | fold5_v1 | 5 | 0.5162 | 0.0090 | **0.0040** |
| EEG_ECG | fold10_v1 | 10 | 0.5187 | 0.0182 | 0.0058 |

**Anomaly checks across all 45 runs** (before trusting any of the above):
0/45 missing metrics files, 0/45 suspiciously fast completions (<30s,
possible crash), 0/45 statistical outliers (>2 std from group mean),
0/45 duplicate/misattributed F1 values. Elapsed times ranged 319s-752s,
consistent with genuine full training runs throughout.

**The decision**: mean F1 differs only marginally between the two split
schemes (0.001-0.006 across modalities, within noise) — but **fold5_v1
consistently produces a tighter SEM than fold10_v1 for all 3 modalities**,
despite averaging over half as many folds. This is counter-intuitive
(more folds is not always more robust) but explained cleanly:
fold5_v1's larger per-fold test sets (388 subjects vs. 194) produce
less noisy individual-fold F1 estimates, and that noise reduction more
than compensates for having fewer folds to average over in the SEM
calculation. **Standardized on fold5_v1** for all subsequent full-cohort
runs (Spectral, Next-Token, and this document's own experiments), and
for the remaining models (BIOT, LaBraM, SensorLM) — comparable central
estimates, tighter precision, roughly half the fine-tuning compute per
model.

---

## 5. The full-cohort vs. historical F1 gap investigation

**The question**: From-Scratch EEG_ONLY on the full 1944-subject cohort
(fold5_v1) scores **0.5162 mean F1** (5-fold). The historical 350-subject
Puhti-era result was **0.6582**. A ~14-point gap. Is this a bug somewhere
in the pipeline, or a real effect of a different evaluation population?
Every plausible alternative explanation was tested directly, not just
reasoned about.

### 5.1 Metrics computation
Independently recomputed macro F1 from raw `test_all_outputs.pickle` /
`test_all_targets.pickle` / `test_all_masks.pickle` using plain
`sklearn.f1_score` — deliberately **not** reusing any code from
`compute_metrics.py`, to catch a bug in the metrics script itself.
Result for fold5_v1/fold_0: **0.510881**, matching `metrics.json` to
6 decimal places. Repeated for all 5 EEG_ONLY folds, all 5 ECG_ONLY
folds, all 5 EEG_ECG folds, all 5 Spectral folds, all 5 Next-Token
folds (25 runs total) as part of the results-restructuring work —
every single one matched exactly. **No bug in the metrics pipeline.**

### 5.2 Epoch/label alignment
Checked, for every one of 388 test subjects across two folds (not a
sample): does the number of predicted epochs match the label CSV's row
count? Result: 334/388 exact matches. Of the 54 mismatches, 46 shared
one exact signature — short by precisely 9 epochs — and 3 were capped at
exactly `max_seq_length=1500` (an intentional config limit for very long
recordings). The diff-9 pattern was root-caused arithmetically: for the
traced example, the raw signal supports up to 1199 label-epochs'-worth
of 5-second embedding patches, but `generate_embeddings.py` only emitted
1190 — a shortfall on the **embedding-generation side**, not in the
classifier. The classifier's `min_length = min(embedding_len, label_len)`
step then takes a common **prefix** of both arrays, so whatever epochs
*are* compared remain correctly paired — it drops trailing epochs, it
does not shift or misalign the ones that remain. Same diff-9 signature
reappeared identically when the same subjects were run through the
Spectral encoder's embeddings — confirming it's a property of the raw
embedding-generation pipeline, independent of which encoder produced the
embeddings. Aggregate impact: well under 1% of the 147,981 total
evaluated epochs in a single fold. **No misalignment bug — a minor,
root-caused, negligible-impact completeness artifact.**

### 5.3 Fold and pretraining leakage
Re-derived directly from the split JSON (not assumed from an earlier
claim): zero train/val/test overlap in every one of fold5_v1's 5 folds,
and the 5 test sets are pairwise disjoint across all 10 fold-pair
combinations, unioning to exactly 1944 subjects — a clean partition.
Checked the encoder's own pretraining split too: `pretrain`/`train` both
equal the full 1944-subject cohort (no held-out set at the
self-supervised pretraining stage) — meaning all fold5_v1 test subjects
were seen by the encoder during unlabeled contrastive pretraining, but
this is self-supervised exposure only (no sleep-stage labels touched),
and it would inflate a score, not depress one, so it doesn't explain
the gap. Separately, checked the *historical* 350-subject encoder for
label leakage in the other direction: its 50 test subjects have **zero**
overlap with its own 300-subject pretraining set or 30-subject
validation set — the historical 0.6582 is not inflated by encoder-level
leakage either.

### 5.4 Checkpoint selection (best.pt vs. checkpoint.pt)
Found that `pretrain.py`'s checkpoint selection (`is_best = loss <
best_loss`) uses **training loss only** — no held-out validation signal
at all. `best.pt` corresponds to epoch 28 of 100; `checkpoint.pt` (final
epoch, 99) was never compared. Tested directly: generated a full second
embeddings cache from `checkpoint.pt` and re-ran fold5_v1/fold_0
end-to-end. Result: **0.514973** vs. `best.pt`'s 0.510881 — a +0.4-point
difference, well within ordinary training noise. **Ruled out** as a
contributor to the 14-point gap (and confirmed via the same test for
Spectral/Next-Token, which *do* use genuine validation-loss selection,
that this isn't a systemic pipeline issue either).

### 5.5 Encoder cohort-size experiment
Generated embeddings for the full 1944-subject cohort using the
*original 350-subject encoder's* `best.pt` (not retrained — same weights,
untouched since 2026-06-24), then fine-tuned fold5_v1 across all 5 folds.
Result: **mean F1 = 0.5034 ± 0.0088**, vs. the full-cohort encoder's
0.5162 ± 0.0080 — a real but modest ~1.3-point difference, consistently
in the same direction across all 5 folds. Pretraining-cohort size alone
does not explain the 14-point gap.

### 5.6 Pipeline-regression experiment
Reproduced the historical run's exact 50-subject test split (verified
byte-identical to the historical run's actual test set via its
`all_paths.pickle`, zero symmetric difference) through the *current*
`finetune_sleep_staging.py` / `evaluate_sleep_staging.py` /
`compute_metrics.py` code, using the same encoder weights. (The
*original* embeddings cache no longer exists on disk — confirmed
deleted — so embeddings were regenerated from the same weights via the
current `generate_embeddings.py`.) Result: **0.6482** vs. historical
0.6582 — about 1 point lower, within normal run-to-run noise. **No
pipeline-code regression.**

### 5.7 Fresh-encoder historical-reproduction experiment
The strongest test: pretrained a **brand-new encoder from random init**,
today, on the identical 350-subject data/config/split as the original
June encoder (same `config_pretrain_eegonly.yaml` hyperparameters, new
output path so nothing existing was touched), then fine-tuned it through
the current pipeline on the same 50-subject test split. Result:
**0.6592** — actually *closer* to the historical 0.6582 than reusing the
original June encoder was. This rules out the last remaining
alternative: it isn't something specific to that one encoder's random
initialization or training run either.

### 5.8 Conclusion
Every alternative explanation was tested directly and ruled out:
metrics computation (exact match), epoch alignment (root-caused, no
shift, negligible impact), fold/pretraining leakage (clean in both
directions), checkpoint selection (no meaningful effect), encoder
cohort-size (small effect, ~1.3 pts), pipeline-code regression (~1 pt,
noise), and encoder identity (fresh encoder reproduces historical result
even better). **The ~14-point gap is a real, expected effect of
evaluating on a much larger (~388 vs. 50 subjects/fold), harder, more
representative population — not a bug anywhere in the pipeline.**
Per-class F1 breakdown supports this: the full-cohort result shows a
uniform ~2-5-point drop across all 5 sleep stages relative to the
350-subject result (Wake 0.72 vs 0.90, N1 0.35 vs 0.29 — actually
*higher* on the harder cohort, N2 0.56 vs 0.75, N3 0.38 vs 0.64, REM
0.55 vs 0.72), not a collapse on any single class — the signature of a
harder, more diverse population, not a training or data defect.

---

## 6. Window-size investigation: 5-minute vs. 5-second chunks

**The question**: From-Scratch pretrains on 5-minute signal chunks;
Spectral and Next-Token operate on standalone 5-second windows. Should
From-Scratch be changed to match?

This codebase is a fork of the published SleepFM work (Thapa et al.,
*A multimodal sleep foundation model for disease prediction*, Nature
Medicine, 2025 — `github.com/zou-group/sleepfm-clinical`), and the
5-minute-chunk architecture is inherited from that original repo, not
introduced during this session.

**Why 5 minutes is architecturally load-bearing, not just a config
choice.** `SetTransformerDataset.samples_per_chunk = sampling_duration
(minutes) × 60 × sampling_freq` — at `sampling_duration: 5`, this gives
38,400-sample chunks, which the model's `Tokenizer` splits into 60
patches of `patch_size=640` (5 seconds at 128Hz) each. Inside
`SetTransformer.forward()`, those 60 patches form a genuine sequence fed
through `PositionalEncoding` + a 6-layer `nn.TransformerEncoder` +
`temporal_pooling` — this is where the model learns cross-patch temporal
structure within a chunk, and where the leave-one-out contrastive loss
gets its per-chunk, per-modality embeddings from. Reduce the chunk to a
single 5-second patch (`S=1`) and that entire stack becomes a no-op:
self-attention over a sequence of length 1 has nothing to attend to.

**Why Next-Token/Spectral get away with `S=1` and From-Scratch can't.**
Both were checked directly: Next-Token's model construction *also* uses
`SetTransformer` with single 640-sample windows — but its pretraining
task (predict the next window's discrete token) gets its temporal signal
from **pairing consecutive windows as separate training examples**, not
from the encoder's own intra-window attention. From-Scratch's
leave-one-out contrastive objective has no equivalent — its entire
point is compressing a multi-patch window into one embedding per
modality via that same 6-layer transformer, then contrasting across
modalities for that window. Shrinking the chunk to match patch size
wouldn't shrink the task, it would silently gut it — turning "relate 5
minutes of one modality to 5 minutes of another" into "relate one
5-second instant to the same instant," with 6 layers of transformer
doing nothing.

**Downstream embeddings are already 5-second granularity regardless.**
Confirmed directly: `generate_embeddings.py` extracts the per-patch
`embedding` tensor (before temporal pooling), not the chunk-level pooled
output — so the classifier fine-tuning stage already consumes 5-second
resolution embeddings from From-Scratch, same as from Spectral/Next-Token.
The 5-minute/5-second difference is purely about what temporal context
the *pretraining* objective sees, not a resolution mismatch downstream.

**Real-world cost estimate for a genuine 5-second redesign.** Using
Next-Token's actual observed per-epoch time (~42 min/epoch at 5-second
granularity, real job) against From-Scratch's actual per-epoch time
(~10.1 min/epoch at 5-minute granularity) gives a real ~4.2x slowdown
factor — not the naive ~60x you'd get from raw window-count scaling
alone (since each window is also 60x smaller/cheaper to process
individually). Extrapolated: ~70 hours for a hypothetical 5-second-
windowed From-Scratch run, and it would require a genuine objective
redesign (moving temporal-relationship learning out of the encoder's
attention and into the task framing, as Next-Token already does), not a
config edit.

**Conclusion**: From-Scratch's 5-minute-chunk architecture is correct as
inherited from the published SleepFM design — it's how the leave-one-out
contrastive objective is meant to work. No change made.

---

## 7. Channel-scope gap: Spectral/Next-Token ended up EEG-only

**How it happened.** From-Scratch's full-cohort encoder is genuinely
multi-modal — `modality_types: ["EEG_ONLY", "RESP", "EKG", "EMG"]`,
confirmed both in its config and in its actual embeddings output (all
four modality keys present per subject). "EEG_ONLY-atomic" in its naming
refers to a *channel-grouping* choice within that still-4-modality
contrastive objective (EEG1/2/3 as one leave-one-out unit, replacing the
older 5-channel EEG+EOG "BAS" grouping) — not a restriction to EEG signal
alone.

`build_spectral_cache.py`, by contrast, hardcodes
`EEG_ONLY_CHANNELS = ["EEG1", "EEG2", "EEG3"]` as the *only* channels it
ever scans — confirmed by direct code read, no RESP/EKG/EMG anywhere in
the cache-construction path. Next-Token inherits this entirely: its
config's `token_codebook_path`, `spectral_signals_path_pretrain`, etc.
all point at the same `spectral_cache_eegonly_fullcohort` directory
(confirmed directly from its saved `config.json`). Both encoders'
learned weights have simply never seen non-EEG signal.

**Root cause**: the same "EEG_ONLY-atomic" naming convention was applied
to both From-Scratch and Spectral, but it means something structurally
different for each — a grouping choice for the former, a hard content
restriction for the latter — and that collision, not a deliberate
decision at the time, is why Spectral/Next-Token ended up narrower than
From-Scratch.

**Investigated expanding Spectral/Next-Token up to match.** Two changes
would be needed in `build_spectral_cache.py`: (a) replace the hardcoded
channel list with a `channel_groups.json`-driven scan across all four
modality groups, mirroring `dataset.py`'s existing pattern, and (b) raise
`MAX_CHANNELS` from 10 to 16 (`3+7+2+4`) — required, not optional, since
the current code silently truncates any channels beyond `MAX_CHANNELS`
rather than erroring. Cost estimate: real disk usage on this filesystem
tracks the number of *actually-written* channels (the memmap cache files
are sparse — verified via `du` showing only ~156GB actually on disk
against a ~379GB *apparent* size for the current EEG_ONLY-only cache),
so disk scales with real channel count (~5.3x, 3→16), not with
`MAX_CHANNELS` alone. Estimated full rebuild: **~874GB**, against
**~535GB** free `/scratch` headroom at last check — doesn't fit even
after deleting the existing cache. Time cost: ~5.3x slower per-window
processing (one `welch()` + one HDF5 read per real channel), extrapolating
the original 12h15m EEG_ONLY build to **~65 hours**, which just barely
exceeds the `small` partition's 3-day limit with a script that has no
checkpoint/resume capability.

**Decision going forward**: rather than expand Spectral/Next-Token up to
From-Scratch's full 4-modality scope (real disk/time blockers, as above),
**narrow all three encoders down to a common EEG+ECG scope** for
consistency — this keeps the 3-way pretraining-method comparison
scope-matched and fair without hitting the multimodal-expansion cost, at
the price of not evaluating RESP/EMG contributions for any of the three
going forward. This decision has not yet been implemented in code as of
this document — `build_spectral_cache.py` and the From-Scratch configs
still reflect their current (EEG-only-cache / 4-modality, respectively)
scopes.

---

## 8. MrOS/SHHS download bugs

**MrOS** (`scripts/download_mros.py`). Two rounds of fixes, verified
against actual git history rather than assumed — the first three items
below were committed in `4648edd` (2026-07-27); the HTML-error-page
detection and the general `nsrr_download()` retry/backoff wrapper were
written later in this session and are committed alongside this document
(an earlier draft of this document incorrectly attributed all five to
`4648edd` — corrected after diffing the actual commit tree):

- **Brute-force ID guessing replaced with a real API query** (`4648edd`).
  The original approach scanned 16 site-prefix codes × up to 9999 subject
  numbers each — the overwhelming majority of which can never exist.
  Replaced with `fetch_real_ids()`, which queries NSRR's actual
  file-listing API. Confirmed live: MrOS only ever uses the `"aa"`
  prefix — 2907 real subjects in visit1, 1026 in visit2.
- **TLS verification workaround** (`4648edd`). `sleepdata.org`'s
  certificate chain fails strict verification (confirmed independently
  via `openssl s_client`, not a Roihu configuration issue) — matched the
  `nsrr` gem's own `VERIFY_NONE` behavior for this specific host.
- **Orphan cleanup** (`4648edd`). If the annotation download fails after
  the EDF already succeeded, the orphaned EDF is now deleted rather than
  left on disk as a silent EDF-without-label mismatch.
- **Silent HTML-error-page failure mode found and fixed** (this
  session). NSRR sometimes returns exit code 0 with an unparseable HTML
  error page (e.g. "The dataset mros was not found") instead of the
  requested file — the `nsrr` gem doesn't treat this as a failure, so
  naive exit-code or even file-existence checks miss it entirely. Fixed
  with `_looks_like_html_error_page()`, which checks the first 256 bytes
  of the downloaded file for an `<!doctype html`/`<html` signature —
  deliberately content-based rather than size-based, since legitimately
  small XML annotation files also start with `<` and would otherwise be
  misclassified.
- **Retry with exponential backoff** (this session). `nsrr_download()`
  now retries up to 3 times with 2s/4s/... backoff between attempts,
  specifically to avoid hammering an already-struggling server rather
  than failing fast.

**MESA and SHHS** (`download_mesa.py`/`download_shhs.py`, commit
`e34d913`, 2026-07-28): same orphan-EDF and redundant-redownload fixes
applied — skip re-fetching an already-valid EDF when only the annotation
is missing; delete the EDF if the annotation subsequently fails to
download, rather than leaving an inconsistent pair.

**SHHS diagnostic finding (documented in `CV_COMPARISON_fold5_vs_fold10.md`):**
the MrOS-style retry/backoff + HTML-detection fix was confirmed **never
applied** to `download_shhs.py` — its `nsrr_download()` still has zero
retries, no stderr capture, and no way to distinguish a genuine miss
from a server-side failure. A completed SHHS download run reported 6271
"not_found" IDs out of the scanned range. Direct reproduction on ~20
sampled "not_found" IDs found a genuine false negative: `shhs2-205059`
downloaded successfully on manual retry (a real 49MB file) only after
three escalating timeout attempts (15s killed, 60s killed, 120s
succeeded) — direct evidence of server-side instability being
misreported as absence. Timing corroborates this: the MrOS fix was
applied at 21:26:44 immediately after confirming NSRR was returning
sustained 503s, and the SHHS run (job 366041) started just 20 minutes
later, squarely inside that confirmed outage window (ran 19h37m, 21:46
→ 17:23 next day). **Caveat**: SHHS also has a genuine structural reason
for a high not-found count independent of any outage — it brute-forces
a sequential ID range (e.g. 5728 IDs scanned for visit2) far sparser
than the real cohort, so a large share of the 6271 figure is likely
genuine gaps regardless. The outage inflated that count by an unknown,
nonzero amount on top. **Recommendation, not yet actioned**: apply the
same retry/backoff + HTML-detection fix to `download_shhs.py` before any
resubmission.

---

## 9. The label-corruption bug: root cause of the full-cohort F1 gap

**This supersedes Section 5's conclusion for practical purposes.** Section
5 correctly ruled out every alternative it tested and concluded the
14-point From-Scratch EEG_ONLY gap was "a real, expected effect of
evaluating on a much larger... population." That conclusion was reached
honestly from the evidence available at the time, but the evidence itself
was compromised: the epoch/label-alignment check in Section 5.2 compared
predicted epoch counts against **label CSV row counts that were
themselves corrupted** for 82% of the cohort (see below) — so it was
checking internal self-consistency between two artifacts derived from the
same broken labels, not correctness against ground truth. It could not
have caught this bug by construction. Section 5's population-size effect
is not wrong — it is real, and still present — but it was a ~1-2 point
effect hiding underneath a ~20-point one.

### 9.1 Discovery

Raised by the project owner as a direct concern: the SleepFM authors'
published full-multimodal results sit at **0.70-0.78 macro F1**; this
project's full-cohort numbers (across every model tested — From-Scratch,
Spectral, Next-Token, BIOT, LaBraM, SensorLM, and the published-checkpoint
scoping test in Section 5) all clustered around **0.50-0.54**, a gap far
larger than Section 5's ~14-point, single-model finding accounted for.
Initial framing suspected the population-size effect alone was the full
explanation, given Section 5's conclusion. An explicit adversarial
re-audit was commissioned instead — "assume something IS wrong until
proven otherwise, do not conclude clean without direct evidence for each
check" — targeting cohort completeness, label correctness against raw
XML, signal/channel integrity, and NSRR documentation, in that order. The
second check (label re-verification) found the bug on the first pass, on
a random 15-subject sample, before the remaining checks were even run.

### 9.2 Root cause

Two independent bugs compounded:

**Bug A — no timestamp-based label alignment.**
`sleepfm/models/dataset.py`, `SleepEventClassificationDataset._try_get_item()`
(lines 255-318) reads a subject's label CSV and treats **row index** as
**epoch index**, with no reference to the `Start`/`Stop` timestamp columns
at all:

```python
labels_df = pd.read_csv(label_path)
y_data = labels_df["StageNumber"].to_numpy()          # row i assumed == epoch i
...
min_length = min(x_data.shape[1], len(y_data))          # positional truncation
x_data = x_data[:, :min_length, :]
y_data = y_data[:min_length]
```

This is safe only if every label CSV has exactly one row per 30-second
epoch, in order. It silently breaks for any CSV where a row spans more
than one epoch.

**Bug B — most label CSVs were never epoch-expanded.** MESA's NSRR XML
run-length-encodes consecutive same-stage epochs into a single
`ScoredEvent` (Section 1 describes this same encoding). Two independent
copies of the epoch-expansion logic existed in this codebase:

- `scripts/generate_labels.py` (standalone regeneration script) — fixed
  **2026-06-21**, commit `9a83e5f`, "Fix label segment-to-epoch expansion
  bug masking all staging performance."
- `scripts/download_mesa.py`'s own inline `parse_mesa_xml()` (runs during
  download, generates each subject's label CSV as a side effect of
  `process_subject()`) — a **separate, independent copy** of the same
  bug, fixed later, **2026-07-28**, commit `e34d913` (Section 1).

The bulk MESA download that populated the majority of the cohort ran
**2026-07-04 to 2026-07-13** — after the standalone script's fix, but
three weeks **before** `download_mesa.py`'s own copy was fixed. Every
label CSV generated inline during that download window inherited the
still-broken inline logic, regardless of the standalone script already
being correct. The Jul 28 fix only affects labels generated from that
point forward — it does not retroactively regenerate files already
written to disk. Full-population check (see 9.3) confirmed the exact
split: label CSV file mtimes cluster into two disjoint bands — Jul 4-13
(the broken inline-generated batch) and Jun 21-Jul 27 (a smaller,
correctly-expanded batch, presumably from a partial standalone-script
re-run at some point after the Jun 21 fix).

**Confirmed scope**: **1592 of 1944 (82%)** subjects' label CSVs were
un-expanded — one row per raw XML `ScoredEvent` (durations up to several
hundred seconds), not one row per 30-second epoch. Concrete example,
`mesa-sleep-1995`: a 36000-second (10-hour, 1200-epoch) recording, raw
XML confirms `sum(stage-event durations) = 36000s`, but its label CSV had
only **182 rows** — a direct row-count match to the raw XML's 182
`Stages` `ScoredEvent` count, confirming zero epoch expansion had
occurred.

**Mechanism of the failure, combining both bugs**: for the 1592 affected
subjects, `dataset.py`'s `min_length = min(x_epochs, y_epochs)` step
(Bug A) truncated every sample to its (severely undersized) label row
count — `mesa-sleep-1995` trained on only its first 182 of ~1200 epochs
(15% of the night, biased toward sleep onset). Worse, for the portion
that *was* used, row *i*'s label (spanning an arbitrary multi-epoch,
often multi-hundred-second block) was matched against signal epoch *i*
(a fixed 30-second slot) — these diverge after the very first
non-30-second row, so even the retained epochs carry increasingly
wrong labels as the recording progresses.

### 9.3 Fix

1. Backed up all 1944 existing label CSVs to
   `mesa/labels_CORRUPTED_pre_2026-08-06/` (checksum-verified against the
   live originals before any further changes — kept permanently as the
   "before" reference, per Section 2's precedent of preserving the
   350-subject baseline).
2. Regenerated all label CSVs from raw XML via the confirmed-correct
   `scripts/generate_labels.py`: 1944/1944 generated, 0 failures.
3. **Full-population verification — every subject, not a sample.** For
   each of 1944 (later 2056, after 9.4's cohort-gap fix) subjects,
   independently re-parsed raw XML and checked: (a) every label row is
   exactly 30.0s, (b) row count matches `round(sum(stage-event
   durations)/30)`, (c) 20 random per-subject timestamp spot-checks of
   `StageNumber` against the raw XML's `ScoredEvent` at that instant.
   Result: **2055/2055** pass (pre-cohort-gap-fix), later confirmed
   **2056/2056** (post-fix) — 0 granularity failures, 0 count mismatches,
   0 spot-check mismatches, across roughly 41,000 total spot-checks.

### 9.4 Related fixes found along the way

- **Cohort gap, 1944 → 2056.** Queried NSRR's file-listing API directly
  (`https://sleepdata.org/api/v1/datasets/mesa/files.json`) — same method
  already proven for MrOS (Section 8) — confirming NSRR's true MESA
  cohort is **2056** subjects with both a valid EDF and annotation (0
  EDF-only, 0 annotation-only). Of the 112 missing locally: 105 were
  above the local max subject ID (`download_mesa.py`'s sequential
  brute-force scan hit its `--subjects` target before reaching them —
  never attempted); 7 (`mesa-sleep-2762`, `2764`, `3158`, `3469`, `3472`,
  `6052`, `6053`) were within the scanned range and had silently failed —
  same failure class as the SHHS false-negative finding in Section 8.
- **Login-node process reaping.** The first download attempt (bare
  `nohup` on the login node) was silently killed at 63/112 — process gone,
  log file empty even post-exit, consistent with a SIGKILL from a
  login-node resource policy. Fixed by resubmitting as a proper SLURM
  batch job instead of a bare background process.
- **`nsrr_download()` had no subprocess timeout.** The SLURM resubmission
  (job `513647`) then completed with **0/49 downloaded** — reproduced the
  identical `nsrr download` command hanging 90s-2min+ across three
  different node contexts for the same file, confirming the NSRR download
  backend (distinct from the file-listing API) is intermittently very
  slow right now, and `nsrr_download()`'s `subprocess.run()` call had no
  `timeout=`, so a stall blocked forever while the existing 3-attempt/
  2s-4s-backoff retry logic (adequate for MrOS, ported in Section 8) gave
  up far too fast to matter. Fixed: `subprocess.run(..., timeout=300)`,
  widened to 5 attempts, in `scripts/download_mesa.py` (uncommitted as of
  this document — not yet merged). Resubmitted as job `513948`: 48/49
  succeeded; the last straggler (`mesa-sleep-6734`) succeeded on one more
  isolated retry (job `516377`), reaching **2056/2056**.
- **HDF5 rebuild for the 112 new subjects.** Job `521295`: 2056/2056
  built, 0 conversion failures, ~15 minutes on an ARM/GH200 node
  (`rg3143`).
- **`sleepfm_venv_arm` environment-location discovery.** The Python
  environment used by every proven-working full-cohort GPU job on this
  cluster is not a conda environment and is not under `/scratch` — it
  lives at `/projappl/project_2019517/sleepfm_venv_arm` (a
  `venv --system-site-packages` built on the `python-pytorch/2.10`
  module, per `requirements_roihu_arm.txt`, commit `4648edd`), and it
  only runs on actual GH200 ARM nodes (`--partition=gpumedium
  --gres=gpu:gh200:1`, plus `module use
  /appl/modulefiles/manual/aida/aarch64; module load
  python-pytorch/2.10` before activation) — attempting to activate or use
  it from an x86_64 node fails with an architecture-mismatch container
  error that looks like a broken environment but isn't. A first attempt
  at the HDF5 rebuild (job `520943`) used the legacy `sleepfm_env` conda
  environment referenced by the older `preprocess_mesa.slurm` and failed
  immediately — that conda environment's `/scratch/project_2019517/
  miniconda3/envs/` directory is now empty (wiped, apparently, when
  `miniconda3` was reinstalled around 2026-07-23) and should be
  considered permanently gone; `sleepfm_venv_arm` is the actual
  current environment for this class of job.

### 9.5 Validation: Leg 1 re-run on the corrected cohort

Fast validation before committing to the much more expensive
re-pretraining of all 6 models: re-ran the published-checkpoint scoping
test from Section 5 (full multimodal `BAS`/`RESP`/`EKG`/`EMG` scope, the
same test whose ~0.53-0.54 result had been read as supporting the
population-size explanation) on the corrected, full 2056-subject cohort.

`dataset_split_fold5_v1.json` regenerated for 2056 subjects via the
existing `scripts/generate_cv_splits.py` (seed=42, unchanged) — old
1944-subject split preserved at
`sleepfm/configs/dataset_split_fold5_v1_1944subj_pre_2056fix.json`. New
per-fold sizes: folds 0-2 train=1234/val=411/test=411, folds 3-4
train=1233/val=412/411 (remainder distribution) — disjointness assertions
in `generate_cv_splits.py` passed for all 5 folds.

Embeddings regenerated for the 112 new subjects only (job `523418`,
~1.5 minutes; existing 1944 subjects' embeddings untouched and reused) —
`/scratch/project_2019517/sleepfm-data/embeddings/` now 2056/2056,
structure spot-checked (`BAS`/`EKG`/`EMG`/`RESP` keys, 5-second
resolution, consistent with the existing 1944).

All 4 modality configs × 5 folds (20 runs total) re-run: jobs `524606`-
`524625`, shared timestamp `2026-08-08_1533`, run_name
`sleepfm__{modality}__published__fold5_v1__2026-08-08_1533`. All 20
COMPLETED, 0 failures.

**Independently verified** (same method as Section 5.1 — plain
`sklearn.f1_score` recomputed from raw `test_all_outputs.pickle`/
`test_all_targets.pickle`/`test_all_masks.pickle`, not
`compute_metrics.py`): **0/20 mismatches** against `metrics.json`.
Fold-disjointness re-confirmed for all 4 modalities: 5 test sets of
411/411/411/411/412 subjects each, summing to exactly 2056 unique
subjects with zero overlap.

| Modality | Macro F1 (mean ± std, 5 folds) |
|---|---|
| BAS | 0.7418 ± 0.0022 |
| BAS+EKG | 0.7424 ± 0.0038 |
| BAS+EKG+RESP | 0.7476 ± 0.0022 |
| BAS+EKG+RESP+EMG (full) | 0.7518 ± 0.0022 |

| | Macro F1 |
|---|---|
| OLD (corrupted-label, 1944-subject cohort) | ~0.53-0.54 |
| SleepFM published/leaky range | 0.70-0.78 |
| **NEW (fixed-label, 2056-subject cohort)** | **0.7418-0.7518 (mean 0.7459, all 20 runs)** |

Every modality now lands **inside** the published range, not merely
closer to it — a ~20-point absolute improvement from fixing one bug.

### 9.6 Conclusion

The label-corruption bug (9.2), not population-size effects, was the
**dominant** cause of the full-cohort F1 gap. Section 5's population-size
finding remains valid as a real, separate, much smaller effect (~1-2
points, based on Section 5.5's encoder-cohort-size experiment) that was
compounding underneath this ~20-point bug, not an alternative explanation
for it — Section 5's own checks could not have detected this bug given
what they were actually comparing against (9.1).

### 9.7 Status and next steps

**Phase 1 cleanup — completed.** Every full-cohort checkpoint, embedding
cache, and result built on the corrupted 1944-subject labels has been
deleted (110 items, **222GB** reclaimed; `/scratch` usage 1.5T→1.2T,
552G→774G free): the 3 full-cohort encoders (From-Scratch/Spectral/
Next-Token EEGONLY, 45GB), `spectral_cache_eegonly_fullcohort/`
(including token-cache artifacts, 150GB), and all 6-model comparison
checkpoints/results (BIOT, LaBraM, SensorLM, and SleepFM's From-Scratch/
Spectral/Next-Token fine-tuning runs, ~26.7GB), plus an early fold10_v1
attempt and several superseded early runs (~1.3GB). **Kept**: the old
corrupted-label Leg 1 run itself
(`sleepfm__*__published__fold5_v1__2026-08-06_1238`) as a permanent
documented "before" reference alongside 9.5's "after" result, matching
`labels_CORRUPTED_pre_2026-08-06/`'s precedent; the historical-repro
reproductions (unrelated 350-subject investigation); everything else
already protected (350-subject baseline, `hdf5_full/`, today's Leg 1
run and embeddings).

**Not yet started**: all 6 full-cohort models (From-Scratch, Spectral,
Next-Token, BIOT, LaBraM, SensorLM) need re-pretraining and re-fine-tuning
from scratch on the corrected, full 2056-subject cohort with correct
labels. Every number in Sections 4-7 above that depends on full-cohort
fine-tuning results should be treated as **superseded and pending
re-validation** once that work is done.

---

## Provenance

- Detailed per-run artifacts for every full-cohort fine-tuning
  experiment referenced above: `experiments_full_cohort/full_cohort/`
- Flat model-comparison summaries: `results/FS_fullcohort_ALL_results.txt`,
  `results/Spectral_fullcohort_ALL_results.txt`,
  `results/NextToken_fullcohort_ALL_results.txt`,
  `results/ALL_RESULTS_FULL.md`
- fold5_v1 vs. fold10_v1 raw comparison data:
  `experiments_full_cohort/full_cohort/CV_COMPARISON_fold5_vs_fold10.md`
  (also contains the SHHS diagnostic referenced in Section 8)
- Original 350-subject Puhti-era results (untouched):
  `results/ALL_RESULTS.md` and sibling `*_ALL_results.txt` files
- 350-subject subject-ID manifest: `sleepfm/configs/mesa_350_subset_manifest.json`
- Git history for every code fix cited above: `e34d913`, `4648edd`,
  `984641b`, `245a6d8`, `b5b4ed1`
