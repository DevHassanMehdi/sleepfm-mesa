"""Regression test for check_subject_count() (see sleepfm/utils.py), the
guard against silently dropped subjects/files added after discovering that
every Dataset loader here swallows a missing/unreadable file with a
`continue` and no error. Synthetic split only -- no real checkpoints, no
cluster-specific paths, so this runs anywhere. Plain assertions (no
pytest dependency, matching the rest of this repo).
"""
import os
import sys

SLEEPFM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sleepfm")
sys.path.insert(0, SLEEPFM_DIR)

from utils import check_subject_count  # noqa: E402


def test_raises_when_fewer_loaded_than_split_lists():
    """A synthetic split listing 10 subjects, only 7 actually loaded (3
    dropped by missing files/channels) must raise, naming both counts and
    the directories used."""
    raised = False
    try:
        check_subject_count("test", n_expected=10, n_loaded=7,
                             hdf5_dir="/fake/hdf5", labels_dir="/fake/labels")
    except RuntimeError as e:
        raised = True
        msg = str(e)
        assert "10" in msg
        assert "7" in msg
        assert "3" in msg  # missing count
        assert "/fake/hdf5" in msg
        assert "/fake/labels" in msg
        assert "--allow_missing_subjects" in msg
    assert raised, "expected RuntimeError when fewer subjects loaded than the split lists"
    print("PASS: test_raises_when_fewer_loaded_than_split_lists")


def test_noop_when_counts_match():
    """No shortfall -- must not raise, must not print a warning path."""
    check_subject_count("test", n_expected=10, n_loaded=10, hdf5_dir="/fake/hdf5")
    check_subject_count("test", n_expected=10, n_loaded=12, hdf5_dir="/fake/hdf5")  # more than expected: also fine
    print("PASS: test_noop_when_counts_match")


def test_allow_missing_subjects_suppresses_the_raise():
    """allow_missing=True (the --allow_missing_subjects CLI flag) must let
    a real shortfall through without raising, for deliberate subset runs."""
    check_subject_count("test", n_expected=10, n_loaded=4,
                         hdf5_dir="/fake/hdf5", allow_missing=True)
    print("PASS: test_allow_missing_subjects_suppresses_the_raise")


def main():
    test_raises_when_fewer_loaded_than_split_lists()
    test_noop_when_counts_match()
    test_allow_missing_subjects_suppresses_the_raise()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
