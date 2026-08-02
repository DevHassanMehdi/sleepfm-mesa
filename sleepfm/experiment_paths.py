"""Single source of truth for full-cohort experiment paths.

Both training and eval/metrics code should import ExperimentID (and the
helpers below) from here instead of hand-rolling
os.path.join(...) checkpoint/results paths, so the naming scheme only
exists in one place.

Naming scheme:
    {model}__{modality}__{pretrain_method}__{split_id}__{timestamp}

    model:           sleepfm | biot | labram | sensorlm
    modality:        EEG_ONLY | ECG_ONLY | EEG_ECG   (canonical form --
                     see canonical_modality() for accepted aliases)
    pretrain_method: fromscratch | spectral | nexttoken | combined |
                     finetuned (free-form string, not a strict enum --
                     new pretraining approaches don't require a code change)
    split_id:        identifies the fold scheme/assignment used. Exact
                     convention TBD; treat as an opaque placeholder for now
                     (e.g. "fold5_v1", "fold10_v1").
    timestamp:       YYYY-MM-DD_HHMM

This only affects NEW writes under checkpoints/full_cohort/ and
experiments_full_cohort/full_cohort/ -- the existing results/ and
checkpoints/ trees (350-subject Puhti-era data) are untouched by anything
in this module. Detailed per-run full-cohort artifacts live under
experiments_full_cohort/full_cohort/ (not results/full_cohort/ -- that
path was retired in favor of flat, human-readable summaries directly in
results/, e.g. FS_fullcohort_ALL_results.txt, ALL_RESULTS_FULL.md).
"""
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SLEEPFM_DATA_ROOT = Path("/scratch/project_2019517/sleepfm-data")
REPO_ROOT = Path(__file__).resolve().parent.parent

# Aliases seen across the existing per-model scripts, mapped to the
# canonical modality name used in the new scheme. Add new aliases here --
# this is the only place modality-name mapping should live.
_MODALITY_ALIASES = {
    "EEG_ONLY": "EEG_ONLY",
    "ECG_ONLY": "ECG_ONLY",
    "EKG": "ECG_ONLY",
    "EEG_ECG": "EEG_ECG",
    "EEG_ONLY_EKG": "EEG_ECG",
}


def canonical_modality(raw: str) -> str:
    """Map a model's own internal modality string to the shared naming scheme."""
    try:
        return _MODALITY_ALIASES[raw]
    except KeyError:
        raise ValueError(
            f"Unknown modality '{raw}' -- add it to _MODALITY_ALIASES in "
            f"sleepfm/experiment_paths.py"
        )


@dataclass(frozen=True)
class ExperimentID:
    model: str
    modality: str
    pretrain_method: str
    split_id: str
    timestamp: str

    @property
    def run_name(self) -> str:
        return (f"{self.model}__{self.modality}__{self.pretrain_method}__"
                f"{self.split_id}__{self.timestamp}")

    @property
    def checkpoint_dir(self) -> Path:
        return SLEEPFM_DATA_ROOT / "checkpoints" / "full_cohort" / self.run_name

    @property
    def results_dir(self) -> Path:
        return REPO_ROOT / "experiments_full_cohort" / "full_cohort" / self.run_name

    def fold_dir(self, fold: int) -> Path:
        return self.checkpoint_dir / f"fold_{fold}"


def new_experiment(model: str, modality: str, pretrain_method: str,
                    split_id: str, timestamp: str = None) -> ExperimentID:
    """Create a new ExperimentID, stamped with the current time unless given.

    Creates both checkpoint_dir and results_dir (empty) immediately so
    both exist from the start of a run, even before any file is written
    into them.
    """
    ts = timestamp or datetime.now().strftime("%Y-%m-%d_%H%M")
    exp = ExperimentID(
        model=model,
        modality=canonical_modality(modality),
        pretrain_method=pretrain_method,
        split_id=split_id,
        timestamp=ts,
    )
    exp.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    exp.results_dir.mkdir(parents=True, exist_ok=True)
    return exp


def experiment_from_checkpoint_dir(checkpoint_dir) -> ExperimentID:
    """Reconstruct an ExperimentID from an existing checkpoints/full_cohort/...
    directory -- e.g. for an evaluate_*.py script that's handed a
    checkpoint dir a training run already created.
    """
    name = Path(checkpoint_dir).name
    parts = name.split("__")
    if len(parts) != 5:
        raise ValueError(
            f"'{name}' doesn't look like an ExperimentID.run_name "
            f"(expected 5 '__'-separated components, got {len(parts)})"
        )
    model, modality, pretrain_method, split_id, timestamp = parts
    return ExperimentID(model=model, modality=modality,
                         pretrain_method=pretrain_method,
                         split_id=split_id, timestamp=timestamp)


def link_checkpoint_and_results(exp: ExperimentID, fold_num: int = None) -> None:
    """Cross-link the two config.json files, once each exists.

    Updates whichever side(s) currently exist without clobbering the rest
    of that file's content, and without requiring the other side to exist
    yet -- safe to call from both the training script (right after it
    saves checkpoint_dir/fold_N/config.json) and the metrics-writing step
    (right after it saves results_dir/fold_N/config.json).

    fold_num=None targets the experiment-level (cross-fold aggregate)
    checkpoint_dir/results_dir directly, for callers like compute_metrics.py
    that summarize across all folds rather than reporting a single fold.
    """
    ckpt_dir = exp.fold_dir(fold_num) if fold_num is not None else exp.checkpoint_dir
    results_dir = (exp.results_dir / f"fold_{fold_num}") if fold_num is not None else exp.results_dir
    ckpt_config_path = ckpt_dir / "config.json"
    results_config_path = results_dir / "config.json"

    if ckpt_config_path.exists():
        with open(ckpt_config_path) as f:
            ckpt_config = json.load(f)
        ckpt_config["results_dir"] = str(results_dir)
        with open(ckpt_config_path, "w") as f:
            json.dump(ckpt_config, f, indent=2)

    if results_config_path.exists():
        with open(results_config_path) as f:
            results_config = json.load(f)
        results_config["checkpoint_dir"] = str(ckpt_dir)
        with open(results_config_path, "w") as f:
            json.dump(results_config, f, indent=2)


def write_metrics_bundle(exp: ExperimentID, fold_num: int, metrics: dict,
                          classification_report_text: str,
                          per_subject_rows: list, config: dict) -> None:
    """Write the four results files (metrics.json, classification_report.txt,
    per_subject_results.csv, config.json) and cross-link with checkpoint_dir.

    fold_num=None writes directly into the experiment-level results_dir --
    for cross-fold aggregate summaries (e.g. compute_metrics.py), which are
    the only writer of that experiment's top-level results and don't collide.

    Any other fold_num writes into results_dir/fold_N/ instead. This
    namespacing matters because results_dir is experiment-level (shared by
    every fold of the same model/modality/split_id/timestamp) -- writing a
    single fold's results directly into results_dir would let concurrently
    running folds overwrite each other's metrics (they commonly share the
    same timestamp when submitted in the same minute).

    per_subject_rows: list of dicts, each with at least
        {model, condition, subject_id, macro_f1, accuracy, n_valid_windows}
    """
    fold_results_dir = (exp.results_dir / f"fold_{fold_num}") if fold_num is not None else exp.results_dir
    fold_checkpoint_dir = exp.fold_dir(fold_num) if fold_num is not None else exp.checkpoint_dir
    fold_results_dir.mkdir(parents=True, exist_ok=True)

    with open(fold_results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(fold_results_dir / "classification_report.txt", "w") as f:
        f.write(classification_report_text)

    if per_subject_rows:
        fieldnames = list(per_subject_rows[0].keys())
        with open(fold_results_dir / "per_subject_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_subject_rows)

    config = dict(config)
    config["checkpoint_dir"] = str(fold_checkpoint_dir)
    with open(fold_results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    link_checkpoint_and_results(exp, fold_num)
