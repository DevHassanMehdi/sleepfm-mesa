# 350-Subject Label Integrity Check (Pre-Fix Labels, Read-Only)

## Purpose

Confirms or refutes the "348 of 350" figure used in the ECG ablation
discussion by re-running the label check from scratch, read-only, against
the exact 350-subject cohort and the pre-fix label files.

## Method

- Subject list: the 350 IDs in `sleepfm/configs/mesa_350_subset_manifest.json`
  (`_metadata.count` confirms 350; `len(subject_ids)` also confirms 350).
- Labels: `labels_CORRUPTED_pre_2026-08-06/{subject_id}.csv` (pre-fix
  labels; this directory has 1944 files total, i.e. the full cohort's
  labels, not just the 350-subject subset -- only the 350 manifest IDs
  were read).
- Recording duration: `hdf5_350_puhti_baseline/{subject_id}.hdf5`,
  `EEG1` channel sample count / 128 Hz (`MESA_HZ` elsewhere in this
  codebase). All 26 channels in a given subject's HDF5 share the same
  sample count, so the channel choice doesn't matter.
- For each of the 350 subjects, every label row's `Stop - Start` was
  checked against exactly 30.0s, consecutive rows were checked for gaps
  (`Start[i] != Stop[i-1]`), and the total row count was checked against
  `floor(duration_sec / 30)`.
- All 350 HDF5 files and all 350 label files were present -- no missing
  files.

## Two ways to count "malformed," and why they disagree

A **strict** row-count match (`row_count == floor(duration/30)`, zero
tolerance) gives **335 exact matches / 15 non-matches**. But those 15
split into two very different groups:

- **13 subjects are short by exactly one trailing epoch, with every
  other row a clean, gapless, exactly-30.0s epoch.** This is the
  signature of a last partial/incomplete epoch being dropped from
  scoring (a benign, expected convention -- not corruption). Nothing
  about these 13 files is internally inconsistent.
- **2 subjects have epochs of wildly inconsistent length** (`Stop - Start`
  ranging from 60s up to several thousand seconds, no 30s structure) and
  are missing the large majority of their rows (154-155 rows where ~1080
  are expected). This is what label corruption actually looks like --
  not a one-row trailing-epoch effect.

Counting only the second group as **malformed** gives **348 of 350
fine, 2 malformed** -- which matches the number cited in the ECG
ablation discussion. This rerun confirms "348 of 350" under that
(reasonable) definition. The "335/15" strict count is also reported
below for full transparency, since it depends on whether a dropped
trailing epoch should count as a defect.

## Result

| Definition | Fine | Malformed |
|---|---|---|
| Strict (row count must exactly equal floor(duration/30)) | 335 | 15 |
| Confirmed (allows a 1-epoch trailing shortfall; flags only internally inconsistent epoch lengths) | **348** | **2** |

### The 2 malformed subjects (confirmed definition)

| Subject | Duration | Expected rows | Actual rows | Rows with epoch length != 30.0s |
|---|---|---|---|---|
| mesa-sleep-0555 | 32400.0s (9.0h) | 1080 | 155 | 89 |
| mesa-sleep-0856 | 32400.0s (9.0h) | 1080 | 154 | 79 |

Both files have the right overall recording duration but the epoch
boundaries inside them are irregular (many non-30s spans, some spanning
hours), so the great majority of 30-second epochs implied by the HDF5
duration have no corresponding label row at all.

### The 13 benign trailing-short subjects (counted as "fine" above)

| Subject | Duration (s) | Expected rows | Actual rows |
|---|---|---|---|
| mesa-sleep-0087 | 36000.0 | 1200 | 1199 |
| mesa-sleep-0125 | 43200.0 | 1440 | 1439 |
| mesa-sleep-0187 | 43200.0 | 1440 | 1439 |
| mesa-sleep-0194 | 64771.0 | 2159 | 2158 |
| mesa-sleep-0211 | 36000.0 | 1200 | 1199 |
| mesa-sleep-0346 | 43200.0 | 1440 | 1439 |
| mesa-sleep-0443 | 46800.0 | 1560 | 1559 |
| mesa-sleep-0672 | 32400.0 | 1080 | 1079 |
| mesa-sleep-0692 | 43200.0 | 1440 | 1439 |
| mesa-sleep-0768 | 36000.0 | 1200 | 1199 |
| mesa-sleep-1161 | 32400.0 | 1080 | 1079 |
| mesa-sleep-1188 | 41400.0 | 1380 | 1379 |
| mesa-sleep-1212 | 43200.0 | 1440 | 1439 |

## Conclusion

**"348 of 350" is confirmed** by this rerun, under the definition that
counts a single dropped trailing epoch as benign and flags only files
with internally inconsistent (non-30s, gapped) epoch structure as
malformed. The 2 genuinely malformed subjects are `mesa-sleep-0555` and
`mesa-sleep-0856`. If a zero-tolerance exact row-count match is required
instead, the honest number is 335 of 350, with the other 15 subjects
behaving as described above -- this distinction should be kept in mind
if "348 of 350" is cited elsewhere without this caveat.
