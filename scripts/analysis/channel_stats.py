"""Statistics for the EEG-channel comparison (results/07).

Core primitives (seeded, deterministic given a seed) used by the
reporting pipeline below: a subject-level cluster bootstrap, a paired
Wilcoxon signed-rank test with rank-biserial effect size, Holm's
step-down p-value correction, a two one-sided test (TOST) equivalence
check built on the same bootstrap CI, and the Friedman test for more
than two related channels.

Subjects, not epochs/windows, are the unit of resampling and testing
throughout: one macro F1 (or accuracy, kappa, per-class F1/recall) per
subject per run, exactly as Part C's per-subject table stores it.
"""
import numpy as np
from scipy import stats as scipy_stats


def cluster_bootstrap_ci(values, n_resamples=2000, seed=0, confidence=0.95, statistic=np.mean):
    """Resample subjects (rows of `values`) with replacement, recompute
    `statistic` each time. Returns (point_estimate, lo, hi, resampled_array).
    Percentile bootstrap, two-sided `confidence` interval."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot[i] = statistic(values[idx])
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(boot, [alpha, 1 - alpha])
    return float(statistic(values)), float(lo), float(hi), boot


def paired_bootstrap_ci(a, b, n_resamples=2000, seed=0, confidence=0.95):
    """Bootstrap CI of the pooled mean difference mean(a) - mean(b),
    resampling the SAME subject indices for both arrays each draw (paired
    resampling, preserving within-subject pairing)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert len(a) == len(b)
    rng = np.random.default_rng(seed)
    n = len(a)
    boot = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot[i] = np.mean(a[idx]) - np.mean(b[idx])
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(boot, [alpha, 1 - alpha])
    return float(np.mean(a) - np.mean(b)), float(lo), float(hi), boot


def rank_biserial(diffs):
    """Matched-pairs rank-biserial correlation for a Wilcoxon signed-rank
    test: (sum of positive ranks - sum of negative ranks) / total rank
    sum, over the non-zero differences. In [-1, 1]; sign follows the sign
    convention of `diffs` (diffs = a - b: positive favors a)."""
    diffs = np.asarray(diffs, dtype=float)
    nz = diffs[diffs != 0]
    if len(nz) == 0:
        return 0.0
    ranks = scipy_stats.rankdata(np.abs(nz))
    r_plus = ranks[nz > 0].sum()
    r_minus = ranks[nz < 0].sum()
    total = r_plus + r_minus
    if total == 0:
        return 0.0
    return float((r_plus - r_minus) / total)


def paired_wilcoxon(a, b):
    """Returns (median_diff, wilcoxon_p, rank_biserial_effect_size).
    a - b is the convention throughout (a favored when positive)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diffs = a - b
    median_diff = float(np.median(diffs))
    if np.all(diffs == 0):
        return median_diff, 1.0, 0.0
    try:
        _, p = scipy_stats.wilcoxon(a, b, zero_method="wilcox")
    except ValueError:
        # all-zero or too few non-zero pairs after dropping ties
        p = 1.0
    eff = rank_biserial(diffs)
    return median_diff, float(p), eff


def holm_adjust(pvalues):
    """Holm step-down adjustment. Returns adjusted p-values in the
    original order of `pvalues`."""
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * pvalues[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def tost_verdict(ci_lo, ci_hi, margin, equivalence_confidence=0.90):
    """TOST equivalence test via a CI-inclusion rule: equivalent if the
    `equivalence_confidence` (e.g. 90%) CI of the difference lies
    entirely within (-margin, +margin). `ci_lo`/`ci_hi` must already be
    the bounds of that (not the 95%) CI -- see tost_from_bootstrap.
    Returns "equivalent" (CI entirely inside the margin), "not
    equivalent" (CI entirely outside the margin on one side, i.e. the
    true difference is confidently larger in magnitude than the margin),
    or "inconclusive" (CI straddles a margin boundary -- this margin
    neither confirms nor rules out equivalence)."""
    if ci_lo > -margin and ci_hi < margin:
        return "equivalent"
    if ci_hi <= -margin or ci_lo >= margin:
        return "not equivalent"
    return "inconclusive"


def tost_from_bootstrap(a, b, margin, n_resamples=2000, seed=0):
    """Convenience: paired bootstrap at 90% confidence, then TOST verdict
    at the given margin. Returns (point_diff, ci90_lo, ci90_hi, verdict)."""
    point, lo, hi, _ = paired_bootstrap_ci(a, b, n_resamples=n_resamples, seed=seed, confidence=0.90)
    verdict = tost_verdict(lo, hi, margin)
    return point, lo, hi, verdict


def friedman(*groups):
    """Friedman test across >=3 related groups (same subjects, same
    order). Returns (statistic, p_value)."""
    stat, p = scipy_stats.friedmanchisquare(*groups)
    return float(stat), float(p)


def class_mean_matches_macro(per_class_values, macro_value, tol=0.0006):
    return abs(np.mean(per_class_values) - macro_value) <= tol


# ---------------------------------------------------------------------------
# Reporting pipeline (results/07, Part D). Everything below consumes the
# per-subject table Part C writes to ~/analysis/eeg_channels/per_subject_
# metrics.csv: one row per (model, condition, fold, subject_id), with
# macro_f1/accuracy/kappa and per-class f1_<Stage>/recall_<Stage> columns.
# Subjects, never epochs or folds, are the unit of comparison throughout.
# ---------------------------------------------------------------------------

STAGES = ["Wake", "N1", "N2", "N3", "REM"]
SINGLE_CHANNEL_CONDITIONS = ["EEG1 only", "EEG2 only", "EEG3 only"]

# Which 3-channel condition is each model's "reference" for single-channel
# comparisons (D.2): SleepFM models use the patience-10 reference (matched
# patience to the single-channel runs); BIOT only has a patience-20
# 3-channel run, so that's its reference instead, labeled as such.
REFERENCE_CONDITION = {
    "SleepFM From-Scratch": ("3-channel, patience 10", "patience 10"),
    "SleepFM Next-Token": ("3-channel, patience 10", "patience 10"),
    "BIOT": ("3-channel benchmark", "patience 20 (BIOT has no patience-10 3-channel run)"),
}

MODELS = ["SleepFM From-Scratch", "SleepFM Next-Token", "BIOT"]


def load_per_subject(csv_path):
    import pandas as pd
    return pd.read_csv(csv_path)


def check_same_2056_subjects(df):
    """Every (model, condition) must cover the same 2056 subjects. Returns
    (ok, report_lines, common_subjects)."""
    lines = []
    ok = True
    subject_sets = {}
    for (model, cond), g in df.groupby(["model", "condition"]):
        s = set(g["subject_id"])
        subject_sets[(model, cond)] = s
        if len(s) != 2056:
            ok = False
            lines.append(f"{model} / {cond}: {len(s)} distinct subjects (expected 2056)")
        dup = g["subject_id"].duplicated().sum()
        if dup:
            ok = False
            lines.append(f"{model} / {cond}: {dup} duplicated subject_id rows")
    all_sets = list(subject_sets.values())
    common = set.intersection(*all_sets) if all_sets else set()
    union = set.union(*all_sets) if all_sets else set()
    if common != union:
        ok = False
        lines.append(f"Subject sets differ across (model, condition): "
                      f"{len(union)} subjects appear somewhere, only {len(common)} appear everywhere")
    if ok:
        lines.append(f"All (model, condition) groups cover the same {len(common)} subjects.")
    return ok, lines, common


def aligned_pair(df, model, cond_a, cond_b, metric="macro_f1", subjects=None):
    """Return (subject_ids, values_a, values_b) aligned by subject_id for
    one model's two conditions. `subjects`, if given, restricts to that set
    (e.g. the common-2056 set from check_same_2056_subjects)."""
    a = df[(df["model"] == model) & (df["condition"] == cond_a)].set_index("subject_id")[metric]
    b = df[(df["model"] == model) & (df["condition"] == cond_b)].set_index("subject_id")[metric]
    common = a.index.intersection(b.index)
    if subjects is not None:
        common = common.intersection(list(subjects))
    common = sorted(common)
    return common, a.loc[common].to_numpy(), b.loc[common].to_numpy()


def part_d1_pooled_vs_fold_mean(df, common_subjects, fold_level_df, n_resamples=2000, seed=0):
    """D.1: pooled macro F1 (cluster bootstrap OVER SUBJECTS, mean of each
    subject's own macro F1 -- the statistic the task's own wording and
    "resample subjects" bootstrap both specify) vs the established 5-fold
    mean+-std of the per-fold, epoch-FLATTENED macro F1 (one macro F1 per
    fold, computed by pooling every subject's valid epochs in that fold
    first -- the same methodology already published in results/02/04/06,
    passed in via `fold_level_df`, e.g. ~/analysis/eeg_channels/
    fold_level_metrics.csv). These are two different statistics by
    construction, not two measurements of the same thing -- see the
    "pooled vs fold-mean" note this function's caller prints."""
    rows = []
    for (model, cond), g in df.groupby(["model", "condition"]):
        g = g[g["subject_id"].isin(common_subjects)]
        values = g["macro_f1"].to_numpy()
        pooled, lo, hi, _ = cluster_bootstrap_ci(values, n_resamples=n_resamples, seed=seed)
        fold_g = fold_level_df[(fold_level_df["model"] == model) & (fold_level_df["condition"] == cond)]
        fold_mean = float(fold_g["macro_f1"].mean())
        fold_std = float(fold_g["macro_f1"].std(ddof=0))
        rows.append(dict(model=model, condition=cond,
                          pooled_macro_f1=pooled, ci95_lo=lo, ci95_hi=hi,
                          fold_mean=fold_mean, fold_std=fold_std,
                          n_subjects=len(values), diff_pooled_minus_foldmean=pooled - fold_mean))
    return rows


def part_d2_paired_within_model(df, common_subjects, n_resamples=2000, seed=0):
    """D.2: all three single-channel pairs, plus each single channel vs
    that model's 3-channel reference, within each model. Holm-adjusted
    across the pairs tested within that model."""
    all_rows = []
    for model in MODELS:
        ref_cond, ref_label = REFERENCE_CONDITION[model]
        pairs = [("EEG1 only", "EEG2 only"), ("EEG1 only", "EEG3 only"), ("EEG2 only", "EEG3 only")]
        for ch in SINGLE_CHANNEL_CONDITIONS:
            pairs.append((ch, ref_cond))
        model_rows = []
        for cond_a, cond_b in pairs:
            subs, a, b = aligned_pair(df, model, cond_a, cond_b, "macro_f1", common_subjects)
            _, acc_a, acc_b = aligned_pair(df, model, cond_a, cond_b, "accuracy", common_subjects)
            _, kap_a, kap_b = aligned_pair(df, model, cond_a, cond_b, "kappa", common_subjects)
            if len(subs) == 0:
                continue
            med_f1, p_f1, eff_f1 = paired_wilcoxon(a, b)
            med_acc, _, _ = paired_wilcoxon(acc_a, acc_b)
            med_kap, _, _ = paired_wilcoxon(kap_a[~np.isnan(kap_a) & ~np.isnan(kap_b)],
                                             kap_b[~np.isnan(kap_a) & ~np.isnan(kap_b)])
            _, ci_lo, ci_hi, _ = paired_bootstrap_ci(a, b, n_resamples=n_resamples, seed=seed)
            model_rows.append(dict(
                model=model, condition_a=cond_a, condition_b=cond_b,
                reference_label=(ref_label if cond_b == ref_cond else ""),
                n_subjects=len(subs),
                median_diff_macro_f1=med_f1, median_diff_accuracy=med_acc, median_diff_kappa=med_kap,
                wilcoxon_p=p_f1, rank_biserial=eff_f1,
                bootstrap_ci95_lo=ci_lo, bootstrap_ci95_hi=ci_hi,
            ))
        pvals = [r["wilcoxon_p"] for r in model_rows]
        holm = holm_adjust(pvals) if pvals else []
        for r, hp in zip(model_rows, holm):
            r["holm_adjusted_p"] = float(hp)
        all_rows.extend(model_rows)
    return all_rows


def part_d3_tost(df, common_subjects, margins=(0.01, 0.02), n_resamples=2000, seed=0):
    """D.3: TOST equivalence at 90% CI for the same pairs as D.2, at each margin."""
    rows = []
    for model in MODELS:
        ref_cond, ref_label = REFERENCE_CONDITION[model]
        pairs = [("EEG1 only", "EEG2 only"), ("EEG1 only", "EEG3 only"), ("EEG2 only", "EEG3 only")]
        for ch in SINGLE_CHANNEL_CONDITIONS:
            pairs.append((ch, ref_cond))
        for cond_a, cond_b in pairs:
            subs, a, b = aligned_pair(df, model, cond_a, cond_b, "macro_f1", common_subjects)
            if len(subs) == 0:
                continue
            for margin in margins:
                point, lo, hi, verdict = tost_from_bootstrap(a, b, margin, n_resamples=n_resamples, seed=seed)
                rows.append(dict(model=model, condition_a=cond_a, condition_b=cond_b,
                                  reference_label=(ref_label if cond_b == ref_cond else ""),
                                  margin=margin, point_diff=point, ci90_lo=lo, ci90_hi=hi, verdict=verdict))
    return rows


def part_d4_friedman(df, common_subjects):
    """D.4: Friedman test across EEG1/EEG2/EEG3 per model, Holm-adjusted
    pairwise Wilcoxon after."""
    friedman_rows = []
    posthoc_rows = []
    for model in MODELS:
        subs1, a, _ = aligned_pair(df, model, "EEG1 only", "EEG2 only", "macro_f1", common_subjects)
        common3 = (set(df[(df["model"] == model) & (df["condition"] == "EEG1 only")]["subject_id"])
                   & set(df[(df["model"] == model) & (df["condition"] == "EEG2 only")]["subject_id"])
                   & set(df[(df["model"] == model) & (df["condition"] == "EEG3 only")]["subject_id"]))
        if common_subjects is not None:
            common3 = common3 & set(common_subjects)
        common3 = sorted(common3)
        vals = {}
        for ch in SINGLE_CHANNEL_CONDITIONS:
            s = df[(df["model"] == model) & (df["condition"] == ch)].set_index("subject_id")["macro_f1"]
            vals[ch] = s.loc[common3].to_numpy()
        stat, p = friedman(vals["EEG1 only"], vals["EEG2 only"], vals["EEG3 only"])
        friedman_rows.append(dict(model=model, n_subjects=len(common3), statistic=stat, p_value=p))

        pairs = [("EEG1 only", "EEG2 only"), ("EEG1 only", "EEG3 only"), ("EEG2 only", "EEG3 only")]
        model_posthoc = []
        for cond_a, cond_b in pairs:
            med, pw, eff = paired_wilcoxon(vals[cond_a], vals[cond_b])
            model_posthoc.append(dict(model=model, condition_a=cond_a, condition_b=cond_b,
                                       n_subjects=len(common3), median_diff=med, wilcoxon_p=pw, rank_biserial=eff))
        holm = holm_adjust([r["wilcoxon_p"] for r in model_posthoc])
        for r, hp in zip(model_posthoc, holm):
            r["holm_adjusted_p"] = float(hp)
        posthoc_rows.extend(model_posthoc)
    return friedman_rows, posthoc_rows


def part_d5_per_class(df, common_subjects, stages):
    """D.5: same paired tests as D.2, but on per-subject RECALL for the
    given stages (two calls: N1/N3/REM, then Wake/N2)."""
    rows = []
    for model in MODELS:
        ref_cond, ref_label = REFERENCE_CONDITION[model]
        pairs = [("EEG1 only", "EEG2 only"), ("EEG1 only", "EEG3 only"), ("EEG2 only", "EEG3 only")]
        for ch in SINGLE_CHANNEL_CONDITIONS:
            pairs.append((ch, ref_cond))
        for stage in stages:
            metric = f"recall_{stage}"
            model_rows = []
            for cond_a, cond_b in pairs:
                a_df = df[(df["model"] == model) & (df["condition"] == cond_a)].set_index("subject_id")[metric]
                b_df = df[(df["model"] == model) & (df["condition"] == cond_b)].set_index("subject_id")[metric]
                common = sorted(set(a_df.index) & set(b_df.index) & set(common_subjects))
                a_vals, b_vals = a_df.loc[common].to_numpy(), b_df.loc[common].to_numpy()
                valid = ~np.isnan(a_vals) & ~np.isnan(b_vals)
                n_dropped = (~valid).sum()
                a_vals, b_vals = a_vals[valid], b_vals[valid]
                if len(a_vals) == 0:
                    continue
                med, p, eff = paired_wilcoxon(a_vals, b_vals)
                model_rows.append(dict(model=model, stage=stage, condition_a=cond_a, condition_b=cond_b,
                                        reference_label=(ref_label if cond_b == ref_cond else ""),
                                        n_subjects=len(a_vals), n_dropped_undefined=int(n_dropped),
                                        median_diff_recall=med, wilcoxon_p=p, rank_biserial=eff))
            holm = holm_adjust([r["wilcoxon_p"] for r in model_rows]) if model_rows else []
            for r, hp in zip(model_rows, holm):
                r["holm_adjusted_p"] = float(hp)
            rows.extend(model_rows)
    return rows


def part_d6_models_vs_each_other(df, common_subjects):
    """D.6: pairwise model comparisons, per condition, on the same subjects."""
    model_pairs = [("BIOT", "SleepFM Next-Token"), ("BIOT", "SleepFM From-Scratch"),
                   ("SleepFM From-Scratch", "SleepFM Next-Token")]
    conditions = SINGLE_CHANNEL_CONDITIONS + ["3-channel, patience 10", "3-channel benchmark"]
    rows = []
    for model_a, model_b in model_pairs:
        for cond in conditions:
            a_df = df[(df["model"] == model_a) & (df["condition"] == cond)].set_index("subject_id")["macro_f1"]
            b_df = df[(df["model"] == model_b) & (df["condition"] == cond)].set_index("subject_id")["macro_f1"]
            if a_df.empty or b_df.empty:
                continue
            common = sorted(set(a_df.index) & set(b_df.index) & set(common_subjects))
            if not common:
                continue
            a_vals, b_vals = a_df.loc[common].to_numpy(), b_df.loc[common].to_numpy()
            med, p, eff = paired_wilcoxon(a_vals, b_vals)
            rows.append(dict(model_a=model_a, model_b=model_b, condition=cond, n_subjects=len(common),
                              median_diff_macro_f1=med, wilcoxon_p=p, rank_biserial=eff))
    return rows


def part_d7_fold_level(df):
    """D.7: per-fold paired differences and sign counts only (no p-value
    beyond noting the n=5 Wilcoxon floor)."""
    rows = []
    for model in MODELS:
        ref_cond, ref_label = REFERENCE_CONDITION[model]
        pairs = [("EEG1 only", "EEG2 only"), ("EEG1 only", "EEG3 only"), ("EEG2 only", "EEG3 only")]
        for ch in SINGLE_CHANNEL_CONDITIONS:
            pairs.append((ch, ref_cond))
        for cond_a, cond_b in pairs:
            a_fold = df[(df["model"] == model) & (df["condition"] == cond_a)].groupby("fold")["macro_f1"].mean()
            b_fold = df[(df["model"] == model) & (df["condition"] == cond_b)].groupby("fold")["macro_f1"].mean()
            common_folds = sorted(set(a_fold.index) & set(b_fold.index))
            if not common_folds:
                continue
            diffs = (a_fold.loc[common_folds] - b_fold.loc[common_folds]).to_numpy()
            rows.append(dict(model=model, condition_a=cond_a, condition_b=cond_b,
                              reference_label=(ref_label if cond_b == ref_cond else ""),
                              n_folds=len(common_folds), per_fold_diff=diffs.tolist(),
                              n_positive=int((diffs > 0).sum()), n_negative=int((diffs < 0).sum()),
                              n_zero=int((diffs == 0).sum())))
    return rows
