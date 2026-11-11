Task 1 — SHHS Diagnostic
Verdict: the 6271 "not_found" figure is NOT fully trustworthy — confirmed via direct reproduction, not just inference.

download_shhs.py's nsrr_download() has the identical pre-fix gap: zero retries, no stderr capture, no distinct logging between a genuine miss and a server-side failure — the exact function I already fixed in download_mros.py, never applied here.

Timing overlap confirmed: download_mros.py's retry/backoff fix was applied at 2026-07-28 21:26:44, right after re-confirming NSRR was still returning sustained 503s. Job 366041 started at 21:46:32 — only 20 minutes later, squarely inside the confirmed outage window (ran 19h37m total, 21:46 Jul 28 → 17:23 Jul 29).

Direct reproduction found a genuine false negative: of ~20 sampled "not_found" IDs (all from visit2/shhs2 — visit1 had zero not_found this run), most returned a clean, fast "0 files" response (plausibly genuine gaps — SHHS's brute-force ID scan covers ranges far sparser than the real cohort size, so many true gaps are expected regardless of any outage). But shhs2-205059 downloaded successfully just now (49MB, real file) — and tellingly, it took three escalating attempts (15s→killed, 60s→killed, 120s→succeeded) for the identical request to finally succeed, which is direct evidence of server-side instability, not genuine absence.

Caveat worth flagging: unlike MrOS, SHHS also has a structural reason for high not_found counts independent of any outage — it brute-forces a sequential ID range (200077-205804 for visit2, 5728 IDs) against a cohort far smaller than that range, so a large chunk of 6271 is likely genuine gaps. The outage inflated that number by an unknown but nonzero amount.

Recommendation: apply the same retry/backoff + HTML/empty-response sanity check fix to download_shhs.py before resubmitting, matching what's already done for MrOS. Not resubmitting per your instruction — diagnostic only.

Task 2 — CV Comparison File
Written to results/full_cohort/CV_COMPARISON_fold5_vs_fold10.md (new file, first creation). All 45 runs passed rigorous anomaly checks (0 missing metrics, 0 fast-exits, 0 statistical outliers, 0 duplicate F1s). Same grid as before, now with SEM comparison formalized: fold5_v1 gives the tighter estimate for all 3 modalities despite half the folds. Recommendation unchanged: standardize on fold5_v1 for the remaining 5 models.# CV Comparison: fold5_v1 vs fold10_v1 (SleepFM From-Scratch, full 1944-subject cohort)

Generated from 45 completed fine-tuning runs (3 modalities x (5+10) fold rotations),
using the fromscratch_eegonly_fullcohort encoder (best.pt, full-cohort pretrained,
contamination-free). All 45 runs passed anomaly checks: no missing metrics files,
no crashes/fast-exits, no statistical outliers (>2 std from group mean), correct
fold attribution, and bidirectional checkpoint<->results cross-links verified.

## Summary Grid

| Modality | Split | N folds | Mean F1 | Std | SEM | Min | Max |
|---|---|---|---|---|---|---|---|
| EEG_ONLY | fold5_v1 | 5 | 0.5162 | 0.0080 | 0.0036 | 0.5030 | 0.5227 |
| EEG_ONLY | fold10_v1 | 10 | 0.5103 | 0.0215 | 0.0068 | 0.4691 | 0.5349 |
| ECG_ONLY | fold5_v1 | 5 | 0.3943 | 0.0109 | 0.0049 | 0.3776 | 0.4051 |
| ECG_ONLY | fold10_v1 | 10 | 0.4006 | 0.0168 | 0.0053 | 0.3764 | 0.4312 |
| EEG_ECG | fold5_v1 | 5 | 0.5162 | 0.0090 | 0.0040 | 0.5067 | 0.5331 |
| EEG_ECG | fold10_v1 | 10 | 0.5187 | 0.0182 | 0.0058 | 0.4833 | 0.5436 |

## Per-fold detail

### EEG_ONLY

- **fold5_v1**: fold0=0.5109, fold1=0.5227, fold2=0.5222, fold3=0.5030, fold4=0.5222
- **fold10_v1**: fold0=0.5145, fold1=0.4819, fold2=0.5096, fold3=0.5332, fold4=0.5058, fold5=0.5277, fold6=0.4691, fold7=0.4958, fold8=0.5349, fold9=0.5309

### ECG_ONLY

- **fold5_v1**: fold0=0.4036, fold1=0.3776, fold2=0.3998, fold3=0.4051, fold4=0.3853
- **fold10_v1**: fold0=0.4312, fold1=0.3764, fold2=0.4008, fold3=0.3881, fold4=0.3974, fold5=0.4080, fold6=0.3860, fold7=0.4251, fold8=0.4070, fold9=0.3861

### EEG_ECG

- **fold5_v1**: fold0=0.5159, fold1=0.5104, fold2=0.5067, fold3=0.5147, fold4=0.5331
- **fold10_v1**: fold0=0.5218, fold1=0.4833, fold2=0.5247, fold3=0.5388, fold4=0.5040, fold5=0.5342, fold6=0.4966, fold7=0.5245, fold8=0.5436, fold9=0.5156

## fold5 vs fold10: mean difference and precision comparison

| Modality | Mean diff (fold10 - fold5) | fold5 SEM | fold10 SEM | Tighter estimate |
|---|---|---|---|---|
| EEG_ONLY | -0.0059 | 0.0036 | 0.0068 | fold5_v1 |
| ECG_ONLY | +0.0063 | 0.0049 | 0.0053 | fold5_v1 |
| EEG_ECG | +0.0025 | 0.0040 | 0.0058 | fold5_v1 |

## Anomaly check results

- Missing metrics files: 0/45
- Suspiciously fast completions (<30s, possible crash): 0/45
- Statistical outliers (>2 std from group mean): 0/45
- Duplicate/misattributed F1 values: 0/45
- Elapsed time range: 319s-752s (all consistent with genuine full training runs)

## Recommendation

Mean macro F1 differs only marginally between fold5_v1 and fold10_v1 for all three
modalities (differences of 0.001-0.006, well within noise). However, **fold5_v1
consistently produces a lower standard error of the mean (tighter estimate) than
fold10_v1 across all 3 modalities**, despite averaging over half as many folds --
because fold5_v1's larger per-fold test sets (388 subjects vs 194) produce less
noisy individual-fold F1 estimates, more than compensating for the smaller fold count
in the standard-error calculation. This runs counter to the intuition that more folds
is always more robust.

**Recommendation: standardize on fold5_v1** for the remaining 5 models (BIOT, LaBraM,
SensorLM, Spectral, Next-Token) -- comparable central estimates with tighter precision,
and roughly half the fine-tuning compute per model.
