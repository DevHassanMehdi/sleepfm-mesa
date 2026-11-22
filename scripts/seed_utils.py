"""Shared --seed helpers for scripts/finetune_biot.py, finetune_labram.py,
and finetune_sensorlm.py. Opt-in only: a script that never calls set_seed()
(the --seed flag omitted) behaves exactly as before this module existed.
"""
import random
import subprocess

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Pass as DataLoader(..., worker_init_fn=seed_worker) alongside a
    seeded generator= so each DataLoader worker process gets its own
    deterministic RNG state instead of inheriting an identical fork of
    whatever state the parent process happened to be in."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_git_commit(repo_root):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


RANDOMNESS_SOURCES = (
    "Randomness sources and what --seed controls: "
    "(1) model head/encoder initialization (freshly-initialized layers at "
    "model construction) -- controlled by torch.manual_seed + "
    "torch.cuda.manual_seed_all; "
    "(2) training data order (DataLoader shuffle=True) -- controlled by an "
    "explicit per-run torch.Generator seeded from --seed; "
    "(3) dropout masks during training, where the model uses dropout -- "
    "controlled by the CUDA RNG via torch.cuda.manual_seed_all; "
    "(4) DataLoader worker processes (num_workers>0) -- controlled by "
    "worker_init_fn, seeded deterministically per worker. "
    "No data augmentation or random subsampling is used by this script."
)
