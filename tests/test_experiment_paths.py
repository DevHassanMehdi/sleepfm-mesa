import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sleepfm"))

import experiment_paths as ep  # noqa: E402


def _make_exp(root):
    return ep.ExperimentID(model="biot", modality="EEG1_ONLY", pretrain_method="finetuned",
                            split_id="fold5_v1", timestamp="2026-01-01_0000", root=root)


def test_link_checkpoint_and_results_default_writes_checkpoint_side():
    with tempfile.TemporaryDirectory() as d:
        exp = _make_exp(d)
        ckpt_dir = exp.fold_dir(0)
        results_dir = exp.results_dir / "fold_0"
        ckpt_dir.mkdir(parents=True)
        results_dir.mkdir(parents=True)
        (ckpt_dir / "config.json").write_text(json.dumps({"lr": 0.0001}))

        ep.link_checkpoint_and_results(exp, 0)

        ckpt_config = json.loads((ckpt_dir / "config.json").read_text())
        assert ckpt_config["results_dir"] == str(results_dir)
        assert ckpt_config["lr"] == 0.0001


def test_link_checkpoint_and_results_skip_checkpoint_side_leaves_it_untouched():
    with tempfile.TemporaryDirectory() as d:
        exp = _make_exp(d)
        ckpt_dir = exp.fold_dir(0)
        results_dir = exp.results_dir / "fold_0"
        ckpt_dir.mkdir(parents=True)
        results_dir.mkdir(parents=True)
        original_text = json.dumps({"lr": 0.0001})
        (ckpt_dir / "config.json").write_text(original_text)

        before_mtime = (ckpt_dir / "config.json").stat().st_mtime_ns
        ep.link_checkpoint_and_results(exp, 0, skip_checkpoint_side=True)
        after_text = (ckpt_dir / "config.json").read_text()
        after_mtime = (ckpt_dir / "config.json").stat().st_mtime_ns

        assert after_text == original_text
        assert after_mtime == before_mtime


def test_link_checkpoint_and_results_skip_checkpoint_side_still_writes_results_side():
    with tempfile.TemporaryDirectory() as d:
        exp = _make_exp(d)
        ckpt_dir = exp.fold_dir(0)
        results_dir = exp.results_dir / "fold_0"
        ckpt_dir.mkdir(parents=True)
        results_dir.mkdir(parents=True)
        (results_dir / "config.json").write_text(json.dumps({"overall_macro_f1": 0.7}))

        ep.link_checkpoint_and_results(exp, 0, skip_checkpoint_side=True)

        results_config = json.loads((results_dir / "config.json").read_text())
        assert results_config["checkpoint_dir"] == str(ckpt_dir)


def test_write_metrics_bundle_default_touches_checkpoint_config():
    with tempfile.TemporaryDirectory() as d:
        exp = _make_exp(d)
        ckpt_dir = exp.fold_dir(0)
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "config.json").write_text(json.dumps({"lr": 0.0001}))

        ep.write_metrics_bundle(exp, 0, metrics={"overall_macro_f1": 0.7},
                                 classification_report_text="report",
                                 per_subject_rows=[], config={"lr": 0.0001})

        ckpt_config = json.loads((ckpt_dir / "config.json").read_text())
        assert "results_dir" in ckpt_config


def test_write_metrics_bundle_no_checkpoint_link_never_touches_checkpoint_folder():
    with tempfile.TemporaryDirectory() as d:
        exp = _make_exp(d)
        ckpt_dir = exp.fold_dir(0)
        ckpt_dir.mkdir(parents=True)
        original_text = json.dumps({"lr": 0.0001})
        (ckpt_dir / "config.json").write_text(original_text)
        (ckpt_dir / "best.pth").write_text("fake checkpoint weights")
        before = {p.name: p.read_text() for p in ckpt_dir.iterdir()}

        ep.write_metrics_bundle(exp, 0, metrics={"overall_macro_f1": 0.7},
                                 classification_report_text="report",
                                 per_subject_rows=[{"model": "biot", "condition": "EEG1_ONLY",
                                                     "subject_id": "s1", "macro_f1": 0.7,
                                                     "accuracy": 0.8, "n_valid_windows": 100}],
                                 config={"lr": 0.0001}, no_checkpoint_link=True)

        after = {p.name: p.read_text() for p in ckpt_dir.iterdir()}
        assert before == after
        # The results side was still written (not a checkpoint folder).
        results_dir = exp.results_dir / "fold_0"
        assert (results_dir / "metrics.json").exists()
        assert (results_dir / "per_subject_results.csv").exists()
