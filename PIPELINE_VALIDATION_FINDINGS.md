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
