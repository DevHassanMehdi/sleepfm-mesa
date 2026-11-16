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


def hodges_lehmann(diffs):
    """Hodges-Lehmann estimator: the median of all pairwise averages
    (x_i + x_j) / 2 for i <= j, over the paired differences. The matched
    estimator for a Wilcoxon signed-rank test (as the median is to a sign
    test), robust and not assuming symmetry/normality. Tests/estimates a
    shift in the TYPICAL (median) subject's difference, same population
    quantity the paired Wilcoxon test is sensitive to -- NOT the same
    target as a mean-based bootstrap CI, which is why both are reported
    rather than one derived from the other."""
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    iu = np.triu_indices(n)
    pairwise_avg = (diffs[iu[0]] + diffs[iu[1]]) / 2.0
    return float(np.median(pairwise_avg))


# ---------------------------------------------------------------------------
# PRIMARY statistic (task item 1): epoch-pooled macro F1. Pool every valid
# epoch (across all subjects selected into a resample, or all subjects in a
# run/fold), build ONE 5x5 confusion matrix from the pooled counts, take
# macro F1 over the 5 fixed classes. This is NOT the same statistic as
# "Mean per-subject macro F1" below (the mean of each subject's own,
# separately-computed macro F1) -- the two differ for the same nonlinear/
# harmonic-mean reason D.1's own note explains, and are kept as two
# clearly-labeled statistics throughout, never conflated.
#
# Needs per-subject CONFUSION-MATRIX counts (not scalar macro_f1 values),
# so that resampling subjects and re-pooling is exact: summing confusion
# matrices over any subset of subjects is exact; summing/averaging
# per-subject macro F1 values is not the same operation.
# ---------------------------------------------------------------------------

def macro_f1_from_cm(cm):
    """Macro F1 over the 5 fixed classes from one pooled 5x5 confusion
    matrix (rows=true, cols=predicted, labels=[0,1,2,3,4])."""
    cm = np.asarray(cm, dtype=float)
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def load_confusion(npz_path):
    """Loads ~/analysis/eeg_channels/per_subject_confusion.npz (written by
    build_confusion.py) into {(model, condition): dict(subjects=, fold=,
    cm=)}. Key sanitization must match build_confusion.py's sanitize()."""
    import numpy as _np
    data = _np.load(npz_path, allow_pickle=True)
    out = {}
    for model in MODELS:
        for cond in SINGLE_CHANNEL_CONDITIONS + ["3-channel, patience 10", "3-channel benchmark"]:
            key = (model + "__" + cond).replace(" ", "_").replace(",", "").replace("-", "_")
            if f"cm__{key}" not in data:
                continue
            out[(model, cond)] = dict(subjects=data[f"subjects__{key}"],
                                       fold=data[f"fold__{key}"],
                                       cm=data[f"cm__{key}"])
    return out


def aligned_cm(confusion_data, model, condition, subjects=None):
    """Returns (sorted_subject_ids, cm_array) for one (model, condition),
    restricted to `subjects` if given, ordered by subject_id."""
    d = confusion_data[(model, condition)]
    idx_by_subject = {s: i for i, s in enumerate(d["subjects"])}
    keys = sorted(idx_by_subject) if subjects is None else sorted(set(idx_by_subject) & set(subjects))
    idx = [idx_by_subject[s] for s in keys]
    return keys, d["cm"][idx]


def aligned_cm_pair(confusion_data, model, cond_a, cond_b, subjects=None):
    """Returns (sorted_subject_ids, cm_a, cm_b), aligned by subject_id, for
    a paired (same-subjects) comparison between two conditions."""
    da = confusion_data[(model, cond_a)]
    db = confusion_data[(model, cond_b)]
    idx_a = {s: i for i, s in enumerate(da["subjects"])}
    idx_b = {s: i for i, s in enumerate(db["subjects"])}
    common = set(idx_a) & set(idx_b)
    if subjects is not None:
        common = common & set(subjects)
    keys = sorted(common)
    return keys, da["cm"][[idx_a[s] for s in keys]], db["cm"][[idx_b[s] for s in keys]]


def pooled_cluster_bootstrap_cm(cm_array, n_resamples=2000, seed=0, confidence=0.95):
    """Resample subjects (rows of cm_array) with replacement, pool
    (sum) their confusion matrices, recompute macro F1 from the pooled
    counts each draw -- exactly the task's "resample subjects ... and
    recompute the pooled macro F1 from pooled counts in each resample."
    Point estimate = macro F1 from the ACTUAL (non-resampled) pooled sum
    over all subjects, not from the bootstrap distribution's own mean."""
    n = cm_array.shape[0]
    point = macro_f1_from_cm(cm_array.sum(axis=0))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot[i] = macro_f1_from_cm(cm_array[idx].sum(axis=0))
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(boot, [alpha, 1 - alpha])
    return point, float(lo), float(hi), boot


def paired_pooled_bootstrap_cm(cm_a, cm_b, n_resamples=2000, seed=0, confidence=0.95):
    """Same subject-resampling indices drawn for both conditions each
    draw (paired, preserving within-subject pairing), pooled macro F1
    difference (a - b) recomputed from pooled counts each draw."""
    assert cm_a.shape[0] == cm_b.shape[0]
    n = cm_a.shape[0]
    point = macro_f1_from_cm(cm_a.sum(axis=0)) - macro_f1_from_cm(cm_b.sum(axis=0))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot[i] = macro_f1_from_cm(cm_a[idx].sum(axis=0)) - macro_f1_from_cm(cm_b[idx].sum(axis=0))
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(boot, [alpha, 1 - alpha])
    return point, float(lo), float(hi), boot


def part_d1_primary_pooled(confusion_data, common_subjects, fold_level_df, n_resamples=2000, seed=0):
    """D.1 (PRIMARY stat): epoch-pooled macro F1 per run (all 2056
    subjects pooled, regardless of fold) with a 95% subject-cluster
    bootstrap CI from pooled counts, next to the per-fold values (already
    epoch-pooled, one confusion matrix per fold -- must equal the
    published 02/04/06 fold macro F1 exactly, carried in fold_level_df)."""
    rows = []
    for (model, cond), d in confusion_data.items():
        keys, cm = aligned_cm(confusion_data, model, cond, common_subjects)
        point, lo, hi, _ = pooled_cluster_bootstrap_cm(cm, n_resamples=n_resamples, seed=seed)
        fold_g = fold_level_df[(fold_level_df["model"] == model) & (fold_level_df["condition"] == cond)]
        rows.append(dict(model=model, condition=cond,
                          run_pooled_macro_f1=point, ci95_lo=lo, ci95_hi=hi,
                          n_subjects=len(keys),
                          per_fold_macro_f1=fold_g.sort_values("fold")["macro_f1"].tolist()))
    return rows


def part_d2_primary_paired(confusion_data, common_subjects, n_resamples=2000, seed=0):
    """D.2 (PRIMARY stat): paired differences on the run-level pooled
    macro F1, same pairs as the secondary D.2, 95% CI via paired pooled
    bootstrap, Holm-adjusted per model. No Wilcoxon p-value here -- the
    pooled statistic is a single number per run, not a per-subject
    sample, so there is nothing to rank-sum; significance is read off the
    bootstrap CI (excludes zero or not), per the task's own wording
    ("paired difference intervals")."""
    all_rows = []
    for model in MODELS:
        ref_cond, ref_label = REFERENCE_CONDITION[model]
        pairs = [("EEG1 only", "EEG2 only"), ("EEG1 only", "EEG3 only"), ("EEG2 only", "EEG3 only")]
        for ch in SINGLE_CHANNEL_CONDITIONS:
            pairs.append((ch, ref_cond))
        for cond_a, cond_b in pairs:
            keys, cm_a, cm_b = aligned_cm_pair(confusion_data, model, cond_a, cond_b, common_subjects)
            if len(keys) == 0:
                continue
            point, lo, hi, _ = paired_pooled_bootstrap_cm(cm_a, cm_b, n_resamples=n_resamples, seed=seed)
            all_rows.append(dict(model=model, condition_a=cond_a, condition_b=cond_b,
                                  reference_label=(ref_label if cond_b == ref_cond else ""),
                                  n_subjects=len(keys), point_diff=point,
                                  ci95_lo=lo, ci95_hi=hi))
    return all_rows


def part_d3_primary_tost(confusion_data, common_subjects, margins=(0.01, 0.02), n_resamples=2000, seed=0):
    """D.3 (PRIMARY stat): TOST equivalence on the run-level pooled macro
    F1 difference, 90% CI via the same paired pooled bootstrap."""
    rows = []
    for model in MODELS:
        ref_cond, ref_label = REFERENCE_CONDITION[model]
        pairs = [("EEG1 only", "EEG2 only"), ("EEG1 only", "EEG3 only"), ("EEG2 only", "EEG3 only")]
        for ch in SINGLE_CHANNEL_CONDITIONS:
            pairs.append((ch, ref_cond))
        for cond_a, cond_b in pairs:
            keys, cm_a, cm_b = aligned_cm_pair(confusion_data, model, cond_a, cond_b, common_subjects)
            if len(keys) == 0:
                continue
            point, lo90, hi90, _ = paired_pooled_bootstrap_cm(cm_a, cm_b, n_resamples=n_resamples,
                                                               seed=seed, confidence=0.90)
            for margin in margins:
                verdict = tost_verdict(lo90, hi90, margin)
                rows.append(dict(model=model, condition_a=cond_a, condition_b=cond_b,
                                  reference_label=(ref_label if cond_b == ref_cond else ""),
                                  margin=margin, point_diff=point, ci90_lo=lo90, ci90_hi=hi90,
                                  verdict=verdict))
    return rows


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
    """D.2 (SECONDARY stat, "Mean per-subject macro F1" -- renamed from
    this project's earlier "pooled macro F1" label, which was really the
    bootstrapped MEAN of each subject's own separately-computed macro F1,
    not an epoch-pooled statistic; see the PRIMARY d1/d2/d3 functions
    above for the actual epoch-pooled statistic now used as this file's
    headline number). Wilcoxon/Friedman and this mean-based bootstrap CI
    test/estimate a shift in the MEAN; the Hodges-Lehmann estimate added
    alongside targets a shift in the TYPICAL (median) subject instead --
    the same population quantity the paired Wilcoxon test is actually
    sensitive to. The two can legitimately disagree in sign or magnitude
    when the per-subject diff distribution is skewed.
    All three single-channel pairs, plus each single channel vs that
    model's 3-channel reference, within each model. Holm-adjusted across
    the pairs tested within that model."""
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
            mean_diff_f1, ci_lo, ci_hi, _ = paired_bootstrap_ci(a, b, n_resamples=n_resamples, seed=seed)
            hl_f1 = hodges_lehmann(a - b)
            model_rows.append(dict(
                model=model, condition_a=cond_a, condition_b=cond_b,
                reference_label=(ref_label if cond_b == ref_cond else ""),
                n_subjects=len(subs),
                mean_diff_macro_f1=mean_diff_f1,
                median_diff_macro_f1=med_f1, hodges_lehmann_diff_macro_f1=hl_f1,
                median_diff_accuracy=med_acc, median_diff_kappa=med_kap,
                wilcoxon_p=p_f1, rank_biserial=eff_f1,
                bootstrap_ci95_lo=ci_lo, bootstrap_ci95_hi=ci_hi,
            ))
        pvals = [r["wilcoxon_p"] for r in model_rows]
        holm = holm_adjust(pvals) if pvals else []
        for r, hp in zip(model_rows, holm):
            r["holm_adjusted_p"] = float(hp)
        all_rows.extend(model_rows)
    return all_rows


def find_significant_but_ci_includes_zero(d2_secondary_rows, alpha=0.05):
    """Task item 2: list every pair where Holm p < alpha but the 95% CI
    of the MEAN difference (mean_diff_macro_f1's bootstrap CI) includes
    zero -- i.e. the rank-based test calls a shift in the typical
    subject significant, but the mean-based CI doesn't rule out a zero
    mean shift. Can happen under skew: many subjects shift one way, a
    few shift hard the other way, pulling the mean's CI toward zero
    while the sign-dominated rank test stays significant."""
    rows = []
    for r in d2_secondary_rows:
        if r["holm_adjusted_p"] < alpha and r["bootstrap_ci95_lo"] <= 0 <= r["bootstrap_ci95_hi"]:
            rows.append(r)
    return rows


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
            hl = hodges_lehmann(vals[cond_a] - vals[cond_b])
            model_posthoc.append(dict(model=model, condition_a=cond_a, condition_b=cond_b,
                                       n_subjects=len(common3), median_diff=med,
                                       hodges_lehmann_diff=hl, wilcoxon_p=pw, rank_biserial=eff))
        holm = holm_adjust([r["wilcoxon_p"] for r in model_posthoc])
        for r, hp in zip(model_posthoc, holm):
            r["holm_adjusted_p"] = float(hp)
        posthoc_rows.extend(model_posthoc)
    return friedman_rows, posthoc_rows


def part_d5_per_class(df, common_subjects, stages, n_resamples=2000, seed=0):
    """D.5 (task item 4): same paired comparisons as D.2, but on
    per-subject RECALL for the given stages (two calls: N1/N3/REM, then
    Wake/N2). Headline is the MEAN paired difference with its bootstrap
    95% CI (never the median -- the median is routinely exactly 0 for a
    stage most subjects tie on, which would be a misleading headline;
    median/Wilcoxon/Holm are kept as the secondary rank-based test).
    `n_subjects` (after dropping undefined) is given for every row."""
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
                mean_diff, ci_lo, ci_hi, _ = paired_bootstrap_ci(a_vals, b_vals, n_resamples=n_resamples, seed=seed)
                model_rows.append(dict(model=model, stage=stage, condition_a=cond_a, condition_b=cond_b,
                                        reference_label=(ref_label if cond_b == ref_cond else ""),
                                        n_subjects=len(a_vals), n_dropped_undefined=int(n_dropped),
                                        mean_diff_recall=mean_diff, bootstrap_ci95_lo=ci_lo, bootstrap_ci95_hi=ci_hi,
                                        median_diff_recall=med, wilcoxon_p=p, rank_biserial=eff))
            holm = holm_adjust([r["wilcoxon_p"] for r in model_rows]) if model_rows else []
            for r, hp in zip(model_rows, holm):
                r["holm_adjusted_p"] = float(hp)
            rows.extend(model_rows)
    return rows


def class_f1_from_cm(cm, c):
    cm = np.asarray(cm, dtype=float)
    tp = cm[c, c]
    fp = cm[:, c].sum() - tp
    fn = cm[c, :].sum() - tp
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


def paired_pooled_bootstrap_class_f1(cm_a, cm_b, c, n_resamples=2000, seed=0, confidence=0.95):
    """Same subject-resampling scheme as paired_pooled_bootstrap_cm, for
    one class's pooled F1 instead of macro F1 -- no subject is ever
    dropped for "undefined": pooling is at the epoch level, so a subject
    with zero occurrences of class c in its own night still contributes
    correctly (0 TP, possibly nonzero FP) to the pooled counts."""
    assert cm_a.shape[0] == cm_b.shape[0]
    n = cm_a.shape[0]
    point = class_f1_from_cm(cm_a.sum(axis=0), c) - class_f1_from_cm(cm_b.sum(axis=0), c)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot[i] = class_f1_from_cm(cm_a[idx].sum(axis=0), c) - class_f1_from_cm(cm_b[idx].sum(axis=0), c)
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(boot, [alpha, 1 - alpha])
    return point, float(lo), float(hi)


def part_d5_pooled_per_class_f1(confusion_data, common_subjects, stages, n_resamples=2000, seed=0):
    """D.5 (task item 4), second half: mean (pooled-count) paired
    difference with a bootstrap 95% CI for POOLED per-class F1 (the
    epoch-pooled F1 for one class, same pooling logic as the primary
    macro-F1 statistic, just not averaged across classes). n given is
    the common-subject count used to build the pooled counts -- never
    fewer, since no subject is dropped for this statistic."""
    rows = []
    for model in MODELS:
        ref_cond, ref_label = REFERENCE_CONDITION[model]
        pairs = [("EEG1 only", "EEG2 only"), ("EEG1 only", "EEG3 only"), ("EEG2 only", "EEG3 only")]
        for ch in SINGLE_CHANNEL_CONDITIONS:
            pairs.append((ch, ref_cond))
        for stage_name in stages:
            c = STAGES.index(stage_name)
            for cond_a, cond_b in pairs:
                keys, cm_a, cm_b = aligned_cm_pair(confusion_data, model, cond_a, cond_b, common_subjects)
                if len(keys) == 0:
                    continue
                point, lo, hi = paired_pooled_bootstrap_class_f1(cm_a, cm_b, c, n_resamples=n_resamples, seed=seed)
                rows.append(dict(model=model, stage=stage_name, condition_a=cond_a, condition_b=cond_b,
                                  reference_label=(ref_label if cond_b == ref_cond else ""),
                                  n_subjects=len(keys), pooled_diff_f1=point,
                                  bootstrap_ci95_lo=lo, bootstrap_ci95_hi=hi))
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


def part_d7_fold_level(df, fold_level_df=None):
    """D.7: per-fold paired differences and sign counts, in TWO forms,
    labeled separately (task item 3): (a) "subject-averaged" -- mean of
    each fold's subjects' own per-subject macro F1 (this function's
    original behavior), and (b) "fold macro F1" -- the epoch-pooled
    per-fold statistic already published in results/02/04/06 (same
    methodology as fold_level_metrics.csv / this file's own D.1 "fold
    values"), passed in via `fold_level_df`. These are two different
    statistics by construction (see D.1's pooled-vs-mean note) and can
    disagree in sign on individual folds."""
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
            row = dict(model=model, condition_a=cond_a, condition_b=cond_b,
                       reference_label=(ref_label if cond_b == ref_cond else ""),
                       n_folds=len(common_folds),
                       per_fold_diff_subject_averaged=diffs.tolist(),
                       n_positive_subject_averaged=int((diffs > 0).sum()),
                       n_negative_subject_averaged=int((diffs < 0).sum()),
                       n_zero_subject_averaged=int((diffs == 0).sum()))
            if fold_level_df is not None:
                fa = fold_level_df[(fold_level_df["model"] == model) & (fold_level_df["condition"] == cond_a)] \
                    .set_index("fold")["macro_f1"]
                fb = fold_level_df[(fold_level_df["model"] == model) & (fold_level_df["condition"] == cond_b)] \
                    .set_index("fold")["macro_f1"]
                common_f2 = sorted(set(fa.index) & set(fb.index))
                fdiffs = (fa.loc[common_f2] - fb.loc[common_f2]).to_numpy()
                row.update(per_fold_diff_fold_macro_f1=fdiffs.tolist(),
                           n_positive_fold_macro_f1=int((fdiffs > 0).sum()),
                           n_negative_fold_macro_f1=int((fdiffs < 0).sum()),
                           n_zero_fold_macro_f1=int((fdiffs == 0).sum()))
            rows.append(row)
    return rows
