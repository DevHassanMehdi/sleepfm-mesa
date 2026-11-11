"""Regression test for rebase_old_home_paths() (see sleepfm/utils.py),
added after an account migration left hundreds of saved checkpoint
config.json files with hardcoded old-account absolute paths. Synthetic
config only -- no real checkpoints, no cluster-specific paths, so this
runs anywhere. Plain assertions (no pytest dependency, matching the rest
of this repo).
"""
import os
import sys

SLEEPFM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sleepfm")
sys.path.insert(0, SLEEPFM_DIR)

from utils import rebase_old_home_paths  # noqa: E402


def test_rebases_top_level_and_nested_old_prefix():
    """Old-prefix strings are rewritten at the top level and inside nested
    dicts/lists, unrelated paths and non-string values are untouched, and
    the new path lands under the current account's home directory."""
    home = os.path.expanduser("~")
    config = {
        "split_path": "/users/hamehdi/projects/sleepfm-mesa/data/mesa/dataset_split_fold5_v1_fold0_123.json",
        "labels_path": "/users/hamehdi/projects/sleepfm-mesa/data/mesa/labels",
        "nested": {
            "results_dir": "/users/hamehdi/projects/sleepfm-mesa/experiments_full_cohort/full_cohort/run/fold_0",
            "list_of_paths": [
                "/users/hamehdi/projects/sleepfm-mesa/data/mesa/hdf5",
                "/scratch/project_2019517/sleepfm-data/mesa/hdf5_full",
            ],
        },
        # Already-current-account path: must pass through unchanged.
        "data_path": "/scratch/project_2019517/sleepfm-data/mesa/hdf5_full",
        # Unrelated path sharing no prefix with the old home dir at all.
        "pretrained_path": "/scratch/project_2019517/BIOT/pretrained-models/EEG-SHHS+PREST-18-channels.ckpt",
        # Non-string values must be returned unchanged, not raise.
        "epochs": 500,
        "early_stopping_patience": 50,
        "use_wandb": False,
        "max_files": None,
    }

    rebased = rebase_old_home_paths(config)

    assert rebased["split_path"] == os.path.join(
        home, "projects/sleepfm-mesa/data/mesa/dataset_split_fold5_v1_fold0_123.json"
    )
    assert rebased["labels_path"] == os.path.join(home, "projects/sleepfm-mesa/data/mesa/labels")
    assert rebased["nested"]["results_dir"] == os.path.join(
        home, "projects/sleepfm-mesa/experiments_full_cohort/full_cohort/run/fold_0"
    )
    assert rebased["nested"]["list_of_paths"][0] == os.path.join(home, "projects/sleepfm-mesa/data/mesa/hdf5")

    # Unaffected paths pass through byte-for-byte.
    assert rebased["nested"]["list_of_paths"][1] == "/scratch/project_2019517/sleepfm-data/mesa/hdf5_full"
    assert rebased["data_path"] == "/scratch/project_2019517/sleepfm-data/mesa/hdf5_full"
    assert rebased["pretrained_path"] == "/scratch/project_2019517/BIOT/pretrained-models/EEG-SHHS+PREST-18-channels.ckpt"

    # Non-string values pass through unchanged (not stringified, not dropped).
    assert rebased["epochs"] == 500
    assert rebased["early_stopping_patience"] == 50
    assert rebased["use_wandb"] is False
    assert rebased["max_files"] is None

    print("PASS: test_rebases_top_level_and_nested_old_prefix")


def test_noop_when_prefix_absent():
    """A config with no old-prefix strings anywhere must come back
    identical (same values), not just "no crash" -- this is the common
    case (most saved configs were never on the old account at all going
    forward) and must have zero effect."""
    config = {
        "data_path": "/scratch/project_2019517/sleepfm-data/mesa/hdf5_full",
        "model_path": "/scratch/project_2019517/sleepfm-data/checkpoints/encoders/fromscratch_eegonly_fullcohort",
        "nested": {"a": ["/scratch/project_2019517/sleepfm-data/mesa/labels", 42]},
    }
    rebased = rebase_old_home_paths(config)
    assert rebased == config
    print("PASS: test_noop_when_prefix_absent")


def main():
    test_rebases_top_level_and_nested_old_prefix()
    test_noop_when_prefix_absent()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
