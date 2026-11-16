import os
import sys

import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "analysis"))

import channel_stats as cs  # noqa: E402


def test_cluster_bootstrap_ci_contains_mean_and_is_seeded():
    rng = np.random.default_rng(1)
    values = rng.normal(0.7, 0.05, size=200)
    point, lo, hi, boot = cs.cluster_bootstrap_ci(values, n_resamples=500, seed=42)
    assert lo < point < hi
    assert abs(point - np.mean(values)) < 1e-9
    # Same seed -> identical resample draws -> identical CI.
    point2, lo2, hi2, _ = cs.cluster_bootstrap_ci(values, n_resamples=500, seed=42)
    assert (point, lo, hi) == (point2, lo2, hi2)
    # Different seed -> (almost certainly) different CI bounds.
    _, lo3, hi3, _ = cs.cluster_bootstrap_ci(values, n_resamples=500, seed=7)
    assert (lo3, hi3) != (lo, hi)


def test_paired_bootstrap_ci_zero_for_identical_arrays():
    a = np.array([0.7, 0.8, 0.6, 0.75, 0.65])
    point, lo, hi, _ = cs.paired_bootstrap_ci(a, a, n_resamples=200, seed=0)
    assert point == 0.0
    assert lo <= 0.0 <= hi


def test_rank_biserial_all_positive_is_one():
    diffs = np.array([0.1, 0.2, 0.05, 0.3])
    assert cs.rank_biserial(diffs) == 1.0


def test_rank_biserial_all_negative_is_minus_one():
    diffs = np.array([-0.1, -0.2, -0.05, -0.3])
    assert cs.rank_biserial(diffs) == -1.0


def test_rank_biserial_zero_diffs_only():
    assert cs.rank_biserial(np.zeros(5)) == 0.0


def test_paired_wilcoxon_detects_consistent_difference():
    rng = np.random.default_rng(3)
    a = rng.normal(0.75, 0.02, size=30)
    b = a - 0.05  # b consistently 0.05 below a, same subjects
    median_diff, p, eff = cs.paired_wilcoxon(a, b)
    assert abs(median_diff - 0.05) < 1e-9
    assert p < 0.01
    assert eff > 0.9


def test_paired_wilcoxon_identical_arrays_gives_p_one():
    a = np.array([0.7, 0.8, 0.6, 0.75, 0.65])
    median_diff, p, eff = cs.paired_wilcoxon(a, a)
    assert median_diff == 0.0
    assert p == 1.0
    assert eff == 0.0


def test_holm_adjust_monotone_and_matches_known_case():
    # Classic example: raw p = [0.01, 0.02, 0.03, 0.04]; Holm multiplies
    # by (n - rank): 4,3,2,1 in sorted order, then enforces monotonicity.
    raw = [0.04, 0.01, 0.03, 0.02]
    adj = cs.holm_adjust(raw)
    sorted_adj = adj[np.argsort(raw)]
    # Non-decreasing once sorted by raw p-value (step-down monotonicity).
    assert all(sorted_adj[i] <= sorted_adj[i + 1] + 1e-12 for i in range(len(sorted_adj) - 1))
    # Smallest raw p (0.01) adjusted by *4, next by *3, etc. before capping at 1.
    assert abs(sorted_adj[0] - 0.04) < 1e-9


def test_holm_adjust_never_exceeds_one():
    adj = cs.holm_adjust([0.5, 0.6, 0.9])
    assert all(0 <= x <= 1 for x in adj)


def test_tost_verdict_equivalent():
    assert cs.tost_verdict(-0.005, 0.006, margin=0.01) == "equivalent"


def test_tost_verdict_not_equivalent():
    assert cs.tost_verdict(-0.03, -0.02, margin=0.01) == "not equivalent"


def test_tost_verdict_inconclusive():
    assert cs.tost_verdict(-0.005, 0.015, margin=0.01) == "inconclusive"


def test_hodges_lehmann_matches_median_for_symmetric_data():
    diffs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert cs.hodges_lehmann(diffs) == 0.0


def test_hodges_lehmann_robust_to_one_outlier():
    diffs = np.array([0.01, 0.02, 0.015, 0.018, 100.0])
    # The single huge outlier should barely move the HL estimate, unlike the mean.
    assert cs.hodges_lehmann(diffs) < 1.0


def test_paired_ttest_matches_scipy():
    rng = np.random.default_rng(5)
    a = rng.normal(0.7, 0.05, size=50)
    b = a - 0.03 + rng.normal(0, 0.01, size=50)
    result = cs.paired_ttest(a, b)
    t_ref, p_ref = scipy_stats.ttest_rel(a, b)
    assert abs(result["t"] - t_ref) < 1e-9
    assert abs(result["p"] - p_ref) < 1e-9
    assert result["df"] == 49
    assert result["ci95_lo"] < result["mean_diff"] < result["ci95_hi"]


def test_paired_ttest_cohens_dz_matches_mean_over_sd():
    a = np.array([0.8, 0.7, 0.9, 0.6, 0.75])
    b = np.array([0.7, 0.65, 0.8, 0.55, 0.70])
    result = cs.paired_ttest(a, b)
    diffs = a - b
    expected_dz = np.mean(diffs) / np.std(diffs, ddof=1)
    assert abs(result["cohens_dz"] - expected_dz) < 1e-9


def test_paired_ttest_zero_for_identical_arrays():
    a = np.array([0.7, 0.8, 0.6, 0.75, 0.65])
    result = cs.paired_ttest(a, a)
    assert result["mean_diff"] == 0.0
    assert result["sd_diff"] == 0.0
    assert result["p"] == 1.0
    assert result["cohens_dz"] == 0.0


def test_skewness_zero_for_symmetric_data():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert abs(cs.skewness(x)) < 1e-9


def test_skewness_positive_for_right_skewed_data():
    x = np.array([1.0, 1.0, 1.0, 1.0, 10.0])
    assert cs.skewness(x) > 0


def test_excess_kurtosis_near_zero_for_normal_like_data():
    rng = np.random.default_rng(9)
    x = rng.normal(0, 1, size=100000)
    assert abs(cs.excess_kurtosis(x)) < 0.2


def test_excess_kurtosis_positive_for_heavy_tailed_data():
    rng = np.random.default_rng(9)
    x = rng.standard_t(df=2, size=10000)  # heavy-tailed
    assert cs.excess_kurtosis(x) > 1.0


def test_kendalls_w_in_unit_range():
    w = cs.kendalls_w(friedman_statistic=20.0, n_subjects=100, k_conditions=3)
    assert 0 <= w <= 1
    assert abs(w - 20.0 / (100 * 2)) < 1e-12


def test_kendalls_w_zero_for_zero_statistic():
    assert cs.kendalls_w(0.0, 100, 3) == 0.0


def test_macro_f1_from_cm_perfect_classifier():
    cm = np.eye(5) * 10
    assert cs.macro_f1_from_cm(cm) == 1.0


def test_macro_f1_from_cm_matches_sklearn():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, size=500)
    y_pred = rng.integers(0, 5, size=500)
    cm = np.zeros((5, 5), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    expected = f1_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average="macro", zero_division=0)
    assert abs(cs.macro_f1_from_cm(cm) - expected) < 1e-9


def test_pooled_cluster_bootstrap_cm_point_is_exact_pooled_value():
    rng = np.random.default_rng(1)
    # 20 subjects, each a 5x5 confusion matrix.
    cm_array = rng.integers(0, 20, size=(20, 5, 5))
    point, lo, hi, boot = cs.pooled_cluster_bootstrap_cm(cm_array, n_resamples=300, seed=0)
    assert point == cs.macro_f1_from_cm(cm_array.sum(axis=0))
    assert lo <= point <= hi
    assert len(boot) == 300


def test_accuracy_from_cm_matches_trace_over_total():
    cm = np.array([[8, 1, 1], [2, 7, 1], [0, 1, 9]])
    assert abs(cs.accuracy_from_cm(cm) - (8 + 7 + 9) / cm.sum()) < 1e-12


def test_accuracy_from_cm_perfect_classifier():
    cm = np.eye(5) * 10
    assert cs.accuracy_from_cm(cm) == 1.0


def test_pooled_cluster_bootstrap_cm_accepts_accuracy_stat_fn():
    rng = np.random.default_rng(1)
    cm_array = rng.integers(0, 20, size=(20, 5, 5))
    point, lo, hi, boot = cs.pooled_cluster_bootstrap_cm(cm_array, n_resamples=300, seed=0,
                                                          stat_fn=cs.accuracy_from_cm)
    assert point == cs.accuracy_from_cm(cm_array.sum(axis=0))
    assert lo <= point <= hi


def test_paired_pooled_bootstrap_cm_zero_for_identical_inputs():
    rng = np.random.default_rng(2)
    cm_array = rng.integers(0, 20, size=(15, 5, 5))
    point, lo, hi, boot = cs.paired_pooled_bootstrap_cm(cm_array, cm_array, n_resamples=200, seed=0)
    assert point == 0.0
    assert lo <= 0.0 <= hi


def test_paired_pooled_bootstrap_cm_detects_a_real_shift():
    rng = np.random.default_rng(3)
    cm_a = rng.integers(5, 25, size=(30, 5, 5))
    # b: same off-diagonal noise, but systematically fewer correct (diagonal) counts.
    cm_b = cm_a.copy()
    cm_b[:, range(5), range(5)] = np.maximum(cm_b[:, range(5), range(5)] - 10, 0)
    point, lo, hi, _ = cs.paired_pooled_bootstrap_cm(cm_a, cm_b, n_resamples=300, seed=0)
    assert point > 0  # a has more correct predictions pooled -> higher macro F1
    assert lo > 0  # a real, large shift should clear the 95% CI


def test_tost_from_bootstrap_equivalent_for_tiny_consistent_diff():
    rng = np.random.default_rng(5)
    a = rng.normal(0.70, 0.01, size=100)
    b = a - 0.001  # tiny, consistent difference well within 0.02 margin
    point, lo, hi, verdict = cs.tost_from_bootstrap(a, b, margin=0.02, n_resamples=500, seed=0)
    assert verdict == "equivalent"


def test_friedman_detects_difference_across_three_groups():
    rng = np.random.default_rng(9)
    a = rng.normal(0.70, 0.01, size=50)
    b = a + 0.10
    c = a + 0.20
    stat, p = cs.friedman(a, b, c)
    assert p < 0.01


def test_class_mean_matches_macro():
    assert cs.class_mean_matches_macro([0.1, 0.2, 0.3, 0.4, 0.5], 0.3, tol=0.0006)
    assert not cs.class_mean_matches_macro([0.1, 0.2, 0.3, 0.4, 0.5], 0.5, tol=0.0006)


# ---------------------------------------------------------------------------
# D-pipeline orchestration, against small synthetic per-subject tables.
# ---------------------------------------------------------------------------

def _synthetic_df(n_subjects=20, seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    subjects = [f"s{i}" for i in range(n_subjects)]
    models = cs.MODELS
    conditions = cs.SINGLE_CHANNEL_CONDITIONS + ["3-channel, patience 10", "3-channel benchmark"]
    rows = []
    folds = {s: i % 5 for i, s in enumerate(subjects)}
    for model in models:
        base = {"SleepFM From-Scratch": 0.70, "SleepFM Next-Token": 0.71, "BIOT": 0.69}[model]
        for cond in conditions:
            # BIOT has no "3-channel, patience 10" run in practice.
            if model == "BIOT" and cond == "3-channel, patience 10":
                continue
            if model != "BIOT" and cond == "3-channel benchmark":
                continue
            bump = 0.05 if "3-channel" in cond else 0.0
            for s in subjects:
                macro_f1 = float(np.clip(base + bump + rng.normal(0, 0.02), 0, 1))
                acc = float(np.clip(macro_f1 + 0.05, 0, 1))
                kappa = float(np.clip(macro_f1 - 0.05, -1, 1))
                row = {"model": model, "condition": cond, "fold": folds[s], "subject_id": s,
                       "n_valid_epochs": 500, "accuracy": acc, "kappa": kappa, "macro_f1": macro_f1}
                for stage in cs.STAGES:
                    row[f"f1_{stage}"] = macro_f1
                    row[f"recall_{stage}"] = macro_f1
                rows.append(row)
    return pd.DataFrame(rows)


def test_check_same_2056_subjects_detects_mismatch():
    df = _synthetic_df(n_subjects=10)
    ok, lines, common = cs.check_same_2056_subjects(df)
    # n_subjects=10 != 2056, so this must report not-ok with a clear reason.
    assert not ok
    assert any("10 distinct subjects" in line for line in lines)


def test_check_same_2056_subjects_passes_when_consistent():
    import pandas as pd
    df = _synthetic_df(n_subjects=5)
    # Patch the expectation by monkey-checking only the "same across groups" part:
    # build a case where every group has the same N (not 2056) and confirm the
    # "subject sets differ" check does NOT fire (only the count-!=2056 one does).
    ok, lines, common = cs.check_same_2056_subjects(df)
    assert len(common) == 5
    assert not any("differ across" in line for line in lines)


def test_aligned_pair_matches_by_subject_id():
    df = _synthetic_df(n_subjects=8)
    subs, a, b = cs.aligned_pair(df, "SleepFM From-Scratch", "EEG1 only", "EEG2 only")
    assert len(subs) == 8
    assert len(a) == len(b) == 8


def _synthetic_fold_level_df(df):
    """A stand-in for the real epoch-flattened fold_level_metrics.csv:
    for this synthetic, IID-per-subject data (no real epochs to pool),
    the per-fold mean of the per-subject values is a reasonable proxy."""
    return df.groupby(["model", "condition", "fold"])["macro_f1"].mean().reset_index()


def test_part_d1_pooled_and_fold_mean_are_close_for_iid_data():
    df = _synthetic_df(n_subjects=40)
    _, _, common = cs.check_same_2056_subjects(df)
    fold_level_df = _synthetic_fold_level_df(df)
    rows = cs.part_d1_pooled_vs_fold_mean(df, common, fold_level_df, n_resamples=200, seed=1)
    assert len(rows) > 0
    for r in rows:
        assert r["ci95_lo"] <= r["pooled_macro_f1"] <= r["ci95_hi"]
        assert abs(r["diff_pooled_minus_foldmean"]) < 0.05


def test_part_d2_detects_the_injected_3channel_bump():
    df = _synthetic_df(n_subjects=40)
    _, _, common = cs.check_same_2056_subjects(df)
    rows = cs.part_d2_paired_within_model(df, common, n_resamples=200, seed=1)
    fs_vs_ref = [r for r in rows if r["model"] == "SleepFM From-Scratch" and r["reference_label"]]
    assert len(fs_vs_ref) == 3  # EEG1/EEG2/EEG3 vs the reference
    for r in fs_vs_ref:
        assert r["median_diff_macro_f1"] < -0.02  # single channel scores lower than the +0.05 reference
        assert "holm_adjusted_p" in r


def test_part_d2_ttest_vs_wilcoxon_detects_the_injected_3channel_bump():
    df = _synthetic_df(n_subjects=40)
    _, _, common = cs.check_same_2056_subjects(df)
    rows = cs.part_d2_ttest_vs_wilcoxon(df, common, metric="macro_f1", n_resamples=200, seed=1)
    fs_vs_ref = [r for r in rows if r["model"] == "SleepFM From-Scratch" and r["reference_label"]]
    assert len(fs_vs_ref) == 3
    for r in fs_vs_ref:
        assert r["mean_diff"] < -0.02
        assert r["ttest_p"] < 0.001
        assert r["ttest_holm_p"] < 0.001
        assert r["ttest_ci95_lo"] <= r["mean_diff"] <= r["ttest_ci95_hi"]
        for k in ("skewness", "excess_kurtosis", "wilcoxon_p", "wilcoxon_holm_p", "hodges_lehmann_diff"):
            assert k in r


def test_part_d2_ttest_vs_wilcoxon_accepts_accuracy_metric():
    df = _synthetic_df(n_subjects=40)
    _, _, common = cs.check_same_2056_subjects(df)
    rows = cs.part_d2_ttest_vs_wilcoxon(df, common, metric="accuracy", n_resamples=200, seed=1)
    assert len(rows) > 0


def test_find_ttest_wilcoxon_disagreements_flags_opposite_significance():
    rows = [
        dict(model="m", condition_a="a", condition_b="b", ttest_holm_p=0.2, wilcoxon_holm_p=0.001),  # disagree
        dict(model="m", condition_a="a", condition_b="c", ttest_holm_p=0.001, wilcoxon_holm_p=0.001),  # agree
        dict(model="m", condition_a="a", condition_b="d", ttest_holm_p=0.2, wilcoxon_holm_p=0.3),  # agree
    ]
    disagreements = cs.find_ttest_wilcoxon_disagreements(rows)
    assert len(disagreements) == 1
    assert disagreements[0]["condition_b"] == "b"


def test_part_fold_level_ttest_low_df_and_detects_bump():
    df = _synthetic_df(n_subjects=40)
    fold_level_df = _synthetic_fold_level_df(df)
    rows = cs.part_fold_level_ttest(fold_level_df, metric="macro_f1")
    fs_vs_ref = [r for r in rows if r["model"] == "SleepFM From-Scratch" and r["reference_label"]]
    assert len(fs_vs_ref) == 3
    for r in fs_vs_ref:
        assert r["df"] == 4  # 5 folds
        assert r["mean_diff"] < -0.02


def test_find_tost_verdict_changes_detects_a_flip():
    primary = [dict(model="m", condition_a="a", condition_b="b", margin=0.01, verdict="inconclusive")]
    secondary = [dict(model="m", condition_a="a", condition_b="b", margin=0.01, verdict="equivalent")]
    changes = cs.find_tost_verdict_changes(primary, secondary)
    assert len(changes) == 1
    assert changes[0]["primary_verdict"] == "inconclusive"
    assert changes[0]["secondary_verdict"] == "equivalent"


def test_find_tost_verdict_changes_empty_when_identical():
    primary = [dict(model="m", condition_a="a", condition_b="b", margin=0.01, verdict="equivalent")]
    secondary = [dict(model="m", condition_a="a", condition_b="b", margin=0.01, verdict="equivalent")]
    assert cs.find_tost_verdict_changes(primary, secondary) == []


def test_part_d3_tost_runs_and_returns_known_verdicts():
    df = _synthetic_df(n_subjects=40)
    _, _, common = cs.check_same_2056_subjects(df)
    rows = cs.part_d3_tost(df, common, margins=(0.01, 0.02), n_resamples=200, seed=1)
    assert len(rows) > 0
    assert all(r["verdict"] in ("equivalent", "not equivalent", "inconclusive") for r in rows)


def test_part_d4_friedman_runs():
    df = _synthetic_df(n_subjects=40)
    _, _, common = cs.check_same_2056_subjects(df)
    friedman_rows, posthoc_rows = cs.part_d4_friedman(df, common)
    assert len(friedman_rows) == len(cs.MODELS)
    assert len(posthoc_rows) == len(cs.MODELS) * 3


def test_part_d5_per_class_tracks_undefined_values():
    df = _synthetic_df(n_subjects=20)
    # Inject a NaN recall for one subject/condition to exercise the drop-count.
    idx = df[(df["model"] == "SleepFM From-Scratch") & (df["condition"] == "EEG1 only")].index[0]
    df.loc[idx, "recall_N1"] = float("nan")
    _, _, common = cs.check_same_2056_subjects(df)
    rows = cs.part_d5_per_class(df, common, ["N1"])
    hit = [r for r in rows if r["model"] == "SleepFM From-Scratch" and r["stage"] == "N1"
           and r["condition_a"] == "EEG1 only"]
    assert hit and hit[0]["n_dropped_undefined"] >= 1


def test_part_d6_models_vs_each_other():
    df = _synthetic_df(n_subjects=40)
    _, _, common = cs.check_same_2056_subjects(df)
    rows = cs.part_d6_models_vs_each_other(df, common)
    assert len(rows) > 0
    assert all("wilcoxon_p" in r for r in rows)


def test_part_d7_fold_level_sign_counts():
    df = _synthetic_df(n_subjects=40)
    rows = cs.part_d7_fold_level(df)
    assert len(rows) > 0
    for r in rows:
        assert r["n_positive_subject_averaged"] + r["n_negative_subject_averaged"] + r["n_zero_subject_averaged"] == r["n_folds"]


def test_part_d7_fold_level_adds_fold_macro_f1_form_when_given():
    df = _synthetic_df(n_subjects=40)
    fold_level_df = _synthetic_fold_level_df(df)
    rows = cs.part_d7_fold_level(df, fold_level_df)
    assert len(rows) > 0
    for r in rows:
        assert "per_fold_diff_fold_macro_f1" in r
        assert (r["n_positive_fold_macro_f1"] + r["n_negative_fold_macro_f1"]
                + r["n_zero_fold_macro_f1"]) == r["n_folds"]
