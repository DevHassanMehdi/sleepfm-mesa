"""Task item 1: rerun every pooled-statistic bootstrap (D.1 point+CI, D.2
paired differences, D.3 TOST, D.5 pooled per-class F1, E6 cross-model) at
10000 resamples under two seeds (0 and 1), and compare against each other
and against the previous 2000-resample/seed-0 result. SEED=0 (10000
resamples) is the one adopted into all tables going forward; SEED=1 is
run only to check stability, never used as a reported value.
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, "/users/hamehd/projects/sleepfm-mesa/scripts/analysis")
import channel_stats as cs

A = os.path.expanduser("~/analysis/eeg_channels")
OUT_DIR = A
N_RESAMPLES = 10000
ADOPTED_SEED = 0       # <-- the seed whose results are used in every table in results/07 and the exports
STABILITY_SEED = 1     # <-- run only to check verdict/interval stability against ADOPTED_SEED, never reported

df = cs.load_per_subject(f"{A}/per_subject_metrics.csv")
fold_level_df = pd.read_csv(f"{A}/fold_level_metrics.csv")
confusion_data = cs.load_confusion(f"{A}/per_subject_confusion.npz")
ok, _, common = cs.check_same_2056_subjects(df)
assert ok

old = json.load(open(f"{A}/part_d_results_v3.json"))  # previous 2000-resample/seed-0 results, for comparison

results = {}
instability_report = []


def compare_tost(label, rows_a, rows_b, src_a, src_b):
    by_key_b = {(r["model"], r["condition_a"], r["condition_b"], r["margin"]): r["verdict"] for r in rows_b}
    n_diff = 0
    for r in rows_a:
        key = (r["model"], r["condition_a"], r["condition_b"], r["margin"])
        vb = by_key_b.get(key)
        if vb is not None and vb != r["verdict"]:
            n_diff += 1
            instability_report.append(dict(check=label, model=r["model"], condition_a=r["condition_a"],
                                            condition_b=r["condition_b"], margin=r["margin"],
                                            **{src_a: r["verdict"], src_b: vb}))
    print(f"  {label}: {n_diff} verdict change(s) of {len(rows_a)}")


for metric, stat_fn in [("macro_f1", cs.macro_f1_from_cm), ("accuracy", cs.accuracy_from_cm)]:
    print(f"\n=== {metric}: D.1/D.2/D.3 primary, {N_RESAMPLES} resamples ===")
    d1_seed0 = cs.part_d1_primary_pooled(confusion_data, common, fold_level_df, n_resamples=N_RESAMPLES,
                                          seed=ADOPTED_SEED, stat_fn=stat_fn, stat_name=metric,
                                          fold_metric="macro_f1" if metric == "macro_f1" else "accuracy")
    d2_seed0 = cs.part_d2_primary_paired(confusion_data, common, n_resamples=N_RESAMPLES, seed=ADOPTED_SEED, stat_fn=stat_fn)
    d3_seed0 = cs.part_d3_primary_tost(confusion_data, common, margins=(0.01, 0.02), n_resamples=N_RESAMPLES,
                                        seed=ADOPTED_SEED, stat_fn=stat_fn)
    d2_seed1 = cs.part_d2_primary_paired(confusion_data, common, n_resamples=N_RESAMPLES, seed=STABILITY_SEED, stat_fn=stat_fn)
    d3_seed1 = cs.part_d3_primary_tost(confusion_data, common, margins=(0.01, 0.02), n_resamples=N_RESAMPLES,
                                        seed=STABILITY_SEED, stat_fn=stat_fn)

    results[f"d1_primary_{metric}"] = d1_seed0
    results[f"d2_primary_{metric}"] = d2_seed0
    results[f"d3_primary_{metric}"] = d3_seed0

    # Stability: seed0(10000) vs seed1(10000)
    compare_tost(f"{metric}: seed0(10k) vs seed1(10k)", d3_seed0, d3_seed1, "seed0_10k", "seed1_10k")
    # Stability: seed0(10000) vs the previous seed0(2000) result
    compare_tost(f"{metric}: seed0(10k) vs seed0(2k, previous)", d3_seed0, old[f"d3_primary_{metric}"],
                 "seed0_10k", "seed0_2k_previous")

    # Print the flagged Next-Token EEG1 vs EEG3 interval at 4 decimals, both seeds, margin 0.01
    for r in d3_seed0:
        if (r["model"] == "SleepFM Next-Token" and r["condition_a"] == "EEG1 only"
                and r["condition_b"] == "EEG3 only" and r["margin"] == 0.01):
            print(f"  {metric} Next-Token EEG1 vs EEG3 @0.01, seed0(10k): diff={r['point_diff']:.4f} "
                  f"CI90=[{r['ci90_lo']:.4f}, {r['ci90_hi']:.4f}] verdict={r['verdict']}")
    for r in d3_seed1:
        if (r["model"] == "SleepFM Next-Token" and r["condition_a"] == "EEG1 only"
                and r["condition_b"] == "EEG3 only" and r["margin"] == 0.01):
            print(f"  {metric} Next-Token EEG1 vs EEG3 @0.01, seed1(10k): diff={r['point_diff']:.4f} "
                  f"CI90=[{r['ci90_lo']:.4f}, {r['ci90_hi']:.4f}] verdict={r['verdict']}")

print(f"\n=== D.5 pooled per-class F1, {N_RESAMPLES} resamples (stability spot-check only reported in log) ===")
for metric_label, stages in [("macro_f1", cs.STAGES)]:
    d5_seed0 = cs.part_d5_pooled_per_class_f1(confusion_data, common, cs.STAGES, n_resamples=N_RESAMPLES, seed=ADOPTED_SEED)
    results["d5_pooled_per_class_f1"] = d5_seed0
    d5_seed1 = cs.part_d5_pooled_per_class_f1(confusion_data, common, cs.STAGES, n_resamples=N_RESAMPLES, seed=STABILITY_SEED)
    # Compare CI-excludes-zero status (the only "verdict"-like property here) between seeds.
    by_key1 = {(r["model"], r["stage"], r["condition_a"], r["condition_b"]):
               (r["bootstrap_ci95_lo"] > 0 or r["bootstrap_ci95_hi"] < 0) for r in d5_seed1}
    n_diff = 0
    for r in d5_seed0:
        key = (r["model"], r["stage"], r["condition_a"], r["condition_b"])
        excl0 = r["bootstrap_ci95_lo"] > 0 or r["bootstrap_ci95_hi"] < 0
        if by_key1.get(key) != excl0:
            n_diff += 1
    print(f"  D.5 pooled per-class F1: {n_diff} CI-excludes-zero flips of {len(d5_seed0)} (seed0 vs seed1, 10k each)")

print(f"\n=== E6 cross-model, {N_RESAMPLES} resamples ===")
e6_pairs = [
    ("BIOT", "EEG3 only", "SleepFM From-Scratch", "EEG3 only", "patience 10", "patience 10"),
    ("BIOT", "3-channel benchmark", "SleepFM From-Scratch", "3-channel benchmark", "patience 20", "patience 50"),
    ("BIOT", "EEG3 only", "SleepFM Next-Token", "EEG3 only", "patience 10", "patience 10"),
    ("BIOT", "3-channel benchmark", "SleepFM Next-Token", "3-channel benchmark", "patience 20", "patience 50"),
]
e6_rows = []
for model_a, cond_a, model_b, cond_b, pat_a, pat_b in e6_pairs:
    da = confusion_data[(model_a, cond_a)]
    db = confusion_data[(model_b, cond_b)]
    idx_a = {s: i for i, s in enumerate(da["subjects"])}
    idx_b = {s: i for i, s in enumerate(db["subjects"])}
    common_ab = sorted(set(idx_a) & set(idx_b))
    cm_a = da["cm"][[idx_a[s] for s in common_ab]]
    cm_b = db["cm"][[idx_b[s] for s in common_ab]]
    point0, lo0, hi0, _ = cs.paired_pooled_bootstrap_cm(cm_a, cm_b, n_resamples=N_RESAMPLES, seed=ADOPTED_SEED)
    point1, lo1, hi1, _ = cs.paired_pooled_bootstrap_cm(cm_a, cm_b, n_resamples=N_RESAMPLES, seed=STABILITY_SEED)
    print(f"  {model_a} ({cond_a}) - {model_b} ({cond_b}): seed0(10k) diff={point0:+.4f} CI=[{lo0:+.4f},{hi0:+.4f}] | "
          f"seed1(10k) diff={point1:+.4f} CI=[{lo1:+.4f},{hi1:+.4f}]")
    e6_rows.append(dict(model_a=model_a, condition_a=cond_a, model_b=model_b, condition_b=cond_b,
                         patience_a=pat_a, patience_b=pat_b, n_subjects=len(common_ab),
                         point_diff=point0, ci95_lo=lo0, ci95_hi=hi0))
results["e6_cross_model"] = e6_rows

print(f"\n=== Instability report: {len(instability_report)} total flips across all checks above ===")
for r in instability_report:
    print(" ", r)

results["instability_report"] = instability_report
results["bootstrap_config"] = dict(n_resamples=N_RESAMPLES, adopted_seed=ADOPTED_SEED, stability_seed=STABILITY_SEED)

with open(os.path.join(OUT_DIR, "part_d_results_v4.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT_DIR}/part_d_results_v4.json")
