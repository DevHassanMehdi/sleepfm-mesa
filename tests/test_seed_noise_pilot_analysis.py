import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "analysis"))

import numpy as np  # noqa: E402
import seed_noise_pilot_analysis as sa  # noqa: E402


FAKE_LOG = """\
RUN: python3 sleepfm/pipeline/finetune_sleep_staging.py --config_path x.yaml --seed 1
E001 loss=0.500 vl=0.400 vf1=0.600 | W=0.9 N1=0.3 N2=0.7 N3=0.6 R=0.7
E002 loss=0.480 vl=0.390 vf1=0.620 | W=0.9 N1=0.3 N2=0.7 N3=0.6 R=0.7 *
E003 loss=0.460 vl=0.385 vf1=0.610 | W=0.9 N1=0.3 N2=0.7 N3=0.6 R=0.7
E012 Early stop | best_f1=0.620 @ epoch 2
RUN: python3 sleepfm/pipeline/evaluate_sleep_staging.py --output_path foo --fold 0 --split test
  run1: macro_f1=0.610000
RUN: python3 sleepfm/pipeline/finetune_sleep_staging.py --config_path x.yaml --seed 2
E001 loss=0.500 vl=0.400 vf1=0.500 | W=0.9 N1=0.3 N2=0.7 N3=0.6 R=0.7
E002 loss=0.480 vl=0.390 vf1=0.530 | W=0.9 N1=0.3 N2=0.7 N3=0.6 R=0.7 *
E013 Early stop | best_f1=0.530 @ epoch 2
RUN: python3 sleepfm/pipeline/evaluate_sleep_staging.py --output_path bar --fold 0 --split test
  run2: macro_f1=0.530000
"""


def test_split_log_into_seed_segments_finds_both_seeds():
    segments = sa.split_log_into_seed_segments(FAKE_LOG)
    assert set(segments) == {1, 2}


def test_split_log_segment_contains_only_its_own_epochs():
    segments = sa.split_log_into_seed_segments(FAKE_LOG)
    assert "E003" in segments[1]
    assert "E003" not in segments[2]
    assert "Early stop | best_f1=0.620" in segments[1]
    assert "Early stop | best_f1=0.530" in segments[2]


def test_parse_epoch_trajectory_extracts_in_order():
    segments = sa.split_log_into_seed_segments(FAKE_LOG)
    traj = sa.parse_epoch_trajectory(segments[1])
    assert traj == [(1, 0.600), (2, 0.620), (3, 0.610)]


def test_parse_stop_and_best_epoch():
    segments = sa.split_log_into_seed_segments(FAKE_LOG)
    stop_epoch, best_epoch, best_f1 = sa.parse_stop_and_best_epoch(segments[1])
    assert (stop_epoch, best_epoch, best_f1) == (12, 2, 0.620)


def test_parse_stop_and_best_epoch_returns_none_when_absent():
    assert sa.parse_stop_and_best_epoch("no such line here") == (None, None, None)


def test_flatness_last_n_epochs_uses_full_trajectory_when_shorter_than_n():
    traj = [(1, 0.5), (2, 0.6), (3, 0.55)]
    n_used, sd, rng = sa.flatness_last_n_epochs(traj, n=30)
    assert n_used == 3
    assert rng == 0.6 - 0.5
    assert sd == np.std([0.5, 0.6, 0.55], ddof=1)


def test_flatness_last_n_epochs_takes_only_the_tail():
    traj = [(i, 0.0) for i in range(1, 21)] + [(21, 1.0), (22, 2.0)]
    n_used, sd, rng = sa.flatness_last_n_epochs(traj, n=2)
    assert n_used == 2
    assert rng == 1.0  # only the last two points (1.0, 2.0), not the leading zeros


def test_flatness_last_n_epochs_empty_trajectory():
    n_used, sd, rng = sa.flatness_last_n_epochs([], n=30)
    assert n_used == 0
    import math
    assert math.isnan(sd) and math.isnan(rng)


def test_chi2_sd_ci_contains_the_sample_sd_and_widens_with_fewer_samples():
    values = [0.68, 0.69, 0.70, 0.67, 0.71]
    sd, lo, hi = sa.chi2_sd_ci(values)
    assert lo < sd < hi
    # a smaller sample should give a wider (or equal) relative CI
    sd2, lo2, hi2 = sa.chi2_sd_ci(values[:3])
    assert (hi2 - lo2) / sd2 >= (hi - lo) / sd


def test_expected_fold_gap_sd_matches_closed_form():
    assert np.isclose(sa.expected_fold_gap_sd(0.01, n_folds=5), 0.01 * np.sqrt(2 / 5))
    assert np.isclose(sa.expected_fold_gap_sd(0.01, n_folds=10), 0.01 * np.sqrt(2 / 10))


def test_percentile_rank_extremes():
    sample = [1, 2, 3, 4, 5]
    assert sa.percentile_rank(0, sample) == 0.0
    assert sa.percentile_rank(6, sample) == 100.0


def test_percentile_rank_matches_membership():
    sample = [1, 2, 3, 4, 5]
    # scipy's "mean" kind: value equal to the 3rd of 5 sorted values ->
    # (count_strictly_below + count_below_or_equal) / 2 / n * 100
    # = (2 + 3) / 2 / 5 * 100 = 50.0
    assert sa.percentile_rank(3, sample) == 50.0
