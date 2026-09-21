"""Shared run-tracking utility for all future training runs (pretraining and
fine-tuning) -- forward-looking only, not used by any already-completed run.

Every run gets one directory:
    runs/{model}/{pretrain_method}/{modality}/{split_id}/[fold_{N}/]{timestamp}/
        tensorboard/        <- TensorBoard event file
        summary.txt         <- single flat text file, written once at the end
        resource_usage.csv  <- nvidia-smi samples, ~30s interval, always-on

fold=None (pretraining -- there is no per-fold concept at that stage) omits
the fold_{N} path component entirely.

The resource sampler runs as a Python-managed subprocess (not a SLURM-level
bash wrapper) so that (a) no SLURM template needs editing -- every script
that calls init_run() gets sampling for free, and (b) a SIGTERM handler here
can write a real "TIMEOUT" summary.txt with whatever epoch/metric state was
last recorded, which a bash-level wrapper has no way to do since it has no
visibility into the Python training loop's state.

Usage:
    from run_tracking import init_run

    tracker = init_run(model="biot", pretrain_method="finetuned",
                        modality="EEG_ONLY", split_id="fold5_v1", fold=0,
                        metric_name="Macro_F1")
    try:
        for epoch in range(...):
            ...
            tracker.log_epoch(epoch + 1, train_metric=train_f1, val_metric=val_f1,
                               per_class_f1=per_class_f1)
            if val_f1 > best_val_f1:
                tracker.mark_best(epoch + 1, val_f1, per_class_f1=per_class_f1)
        tracker.finish("COMPLETED (patience triggered)")  # or "(max epochs)"
    except Exception as e:
        tracker.finish_failed(e)
        raise
"""
import atexit
import csv
import os
import signal
import subprocess
from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parent.parent

CLASS_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


class RunTracker:
    def __init__(self, model, pretrain_method, modality, split_id, fold=None,
                 metric_name="Macro_F1", timestamp=None, sample_interval_s=30):
        self.model = model
        self.pretrain_method = pretrain_method
        self.modality = modality
        self.split_id = split_id
        self.fold = fold
        self.metric_name = metric_name
        self.is_finetune = fold is not None

        # Seconds + PID (not just %H%M) -- two processes for the identical
        # (model, pretrain_method, modality, split_id, fold) tuple starting in
        # the same clock-minute (e.g. a quick FAILED run resubmitted right
        # away, common during debugging) would otherwise collide on the same
        # run_dir and silently overwrite each other's summary.txt/tensorboard/
        # resource_usage.csv. Confirmed by testing: two rapid-fire smoke-test
        # jobs landed in the same directory before this fix.
        self.timestamp = timestamp or f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{os.getpid()}"
        parts = [REPO_ROOT, "runs", model, pretrain_method, modality, split_id]
        if fold is not None:
            parts.append(f"fold_{fold}")
        parts.append(self.timestamp)
        self.run_dir = Path(*parts)
        self.tb_dir = self.run_dir / "tensorboard"
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.run_dir / "summary.txt"
        self.resource_csv_path = self.run_dir / "resource_usage.csv"

        self.writer = SummaryWriter(log_dir=str(self.tb_dir))

        self.start_time = datetime.now()
        self.epochs_run = 0
        self.best_epoch = None
        self.best_metric = None
        self.final_per_class = None
        self._finished = False

        self._sample_interval_s = sample_interval_s
        self._sampler_proc = None
        self._start_resource_sampler()

        # Safety net for process death that skips normal Python control flow
        # (OOM-kill, SIGKILL) -- won't fire on those (nothing can), but does
        # fire for any exit path that isn't an explicit finish() call.
        atexit.register(self._atexit_safety_net)
        self._orig_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._handle_sigterm)

        print(f">>> RUN TRACKING: {self.run_dir}", flush=True)

    # -- resource sampler (B1) -----------------------------------------------
    def _start_resource_sampler(self):
        script = (
            f'echo "timestamp,gpu_util_pct,mem_used_mib,power_draw_w" > "{self.resource_csv_path}"; '
            f'while true; do '
            f'nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw '
            f'--format=csv,noheader,nounits >> "{self.resource_csv_path}" 2>/dev/null; '
            f'sleep {self._sample_interval_s}; done'
        )
        try:
            self._sampler_proc = subprocess.Popen(
                ["bash", "-c", script],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f">>> RUN TRACKING: resource sampler failed to start ({e}) "
                  f"-- continuing without it", flush=True)
            self._sampler_proc = None

    def _stop_resource_sampler(self):
        if self._sampler_proc is not None and self._sampler_proc.poll() is None:
            self._sampler_proc.terminate()
            try:
                # Short timeout deliberately -- this runs AFTER summary.txt is
                # already written, so it's cleanup, not a blocker. A bash loop
                # sleeping on `sleep N` exits on SIGTERM in well under a second;
                # if it doesn't, we don't wait around under a tight kill window.
                self._sampler_proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass  # best-effort; the OS/cgroup will reap it when the job exits
        self._sampler_proc = None

    # -- per-epoch logging (A) -------------------------------------------------
    def log_epoch(self, epoch, train_metric=None, val_metric=None, per_class_f1=None):
        """epoch: 1-indexed, matching every script's existing print convention."""
        self.epochs_run = epoch
        scalars = {}
        if train_metric is not None:
            scalars["train"] = float(train_metric)
        if val_metric is not None:
            scalars["val"] = float(val_metric)
        if scalars:
            self.writer.add_scalars(self.metric_name, scalars, epoch)
        if per_class_f1 is not None:
            self.writer.add_scalars(
                "Val_F1_per_class",
                {name: float(v) for name, v in zip(CLASS_NAMES, per_class_f1)},
                epoch,
            )
        # Refresh summary.txt after every epoch, not just at the end. Testing
        # found SLURM's SIGTERM-to-SIGKILL grace window on this cluster can be
        # short and inconsistent enough that the signal handler's own write
        # sometimes doesn't complete in time -- if that happens, this is the
        # fallback: whatever the last completed epoch wrote stays on disk
        # with accurate epoch/metric state, just possibly a stale "IN_PROGRESS"
        # status instead of "TIMEOUT" for the rare case of a near-zero grace kill.
        self._write_summary("IN_PROGRESS")

    def mark_best(self, epoch, metric, per_class_f1=None):
        self.best_epoch = epoch
        self.best_metric = float(metric)
        if per_class_f1 is not None:
            self.final_per_class = [float(v) for v in per_class_f1]

    # -- completion ------------------------------------------------------------
    def finish(self, status):
        if self._finished:
            return
        self._finished = True
        # Order matters under SIGTERM: SLURM's grace period before SIGKILL can
        # be very short (observed ~8s on this cluster). summary.txt is the
        # one artifact that MUST land, so write it before anything that can
        # block (subprocess teardown, writer flush) risks eating the window.
        self._write_summary(status)
        self._stop_resource_sampler()
        self.writer.close()

    def finish_failed(self, exc):
        self.finish(f"FAILED: {type(exc).__name__}: {exc}")

    def _handle_sigterm(self, signum, frame):
        # SLURM sends SIGTERM ahead of SIGKILL on time-limit -- this is the
        # only way a TIMEOUT run gets a real summary.txt, since the training
        # loop never reaches its own finish() call in that case.
        print(f">>> RUN TRACKING: caught SIGTERM at {datetime.now().isoformat()}, "
              f"writing TIMEOUT summary", flush=True)
        self.finish("TIMEOUT")
        print(">>> RUN TRACKING: TIMEOUT summary written", flush=True)
        signal.signal(signal.SIGTERM, self._orig_sigterm)
        os.kill(os.getpid(), signal.SIGTERM)

    def _atexit_safety_net(self):
        if not self._finished:
            self.finish("FAILED: process exited without an explicit completion "
                        "call (unhandled exception before try/except, or SIGKILL)")

    # -- summary.txt -------------------------------------------------------
    def _read_resource_stats(self):
        if not self.resource_csv_path.exists():
            return None, None
        utils_, mems = [], []
        try:
            with open(self.resource_csv_path) as f:
                reader = csv.reader(f)
                next(reader, None)  # header
                for row in reader:
                    if len(row) < 3:
                        continue
                    try:
                        utils_.append(float(row[1]))
                        mems.append(float(row[2]))
                    except ValueError:
                        continue
        except Exception:
            return None, None
        if not utils_:
            return None, None
        return sum(utils_) / len(utils_), max(mems) / 1024.0

    def _write_summary(self, status):
        finish_time = datetime.now()
        duration = finish_time - self.start_time
        hours, rem = divmod(int(duration.total_seconds()), 3600)
        minutes = rem // 60

        n_gpus = 1
        try:
            import torch
            n_gpus = max(1, torch.cuda.device_count())
        except Exception:
            pass
        gpu_hours = (duration.total_seconds() / 3600.0) * n_gpus

        avg_util, peak_mem_gb = self._read_resource_stats()

        fold_str = f" / fold_{self.fold}" if self.fold is not None else ""
        lines = [
            f"Run: {self.model} / {self.pretrain_method} / {self.modality} / "
            f"{self.split_id}{fold_str}",
            f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Finished: {finish_time.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Duration: {hours}h {minutes}m",
            f"Epochs: {self.epochs_run} | "
            f"Best epoch: {self.best_epoch if self.best_epoch is not None else 'N/A'}",
        ]

        metric_label = "Best val macro F1" if self.is_finetune else f"Best val {self.metric_name}"
        best_metric_str = f"{self.best_metric:.4f}" if self.best_metric is not None else "N/A"
        lines.append(f"{metric_label}: {best_metric_str}")

        if self.is_finetune and self.final_per_class is not None:
            per_class_str = " | ".join(
                f"{name} {v:.4f}" for name, v in zip(CLASS_NAMES, self.final_per_class)
            )
            lines.append(f"Final per-class F1 (val): {per_class_str}")

        lines.append(f"Status: {status}")

        util_str = f"{avg_util:.1f}" if avg_util is not None else "N/A"
        mem_str = f"{peak_mem_gb:.2f}" if peak_mem_gb is not None else "N/A"
        lines.append(
            f"GPU-hours used: {gpu_hours:.3f} | Avg GPU util: {util_str}% | "
            f"Peak mem: {mem_str} GB"
        )

        with open(self.summary_path, "w") as f:
            f.write("\n".join(lines) + "\n")


def init_run(model, pretrain_method, modality, split_id, fold=None,
             metric_name="Macro_F1", timestamp=None):
    return RunTracker(model, pretrain_method, modality, split_id, fold=fold,
                       metric_name=metric_name, timestamp=timestamp)
