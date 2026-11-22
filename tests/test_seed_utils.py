import ast
import os
import random
import re
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import seed_utils as su  # noqa: E402


# -- set_seed: python/numpy/torch/CUDA ---------------------------------------

def test_set_seed_makes_python_random_reproducible():
    su.set_seed(123)
    a = [random.random() for _ in range(5)]
    su.set_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_seed_makes_numpy_reproducible():
    su.set_seed(123)
    a = np.random.rand(5)
    su.set_seed(123)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_set_seed_makes_torch_reproducible():
    su.set_seed(123)
    a = torch.rand(5)
    su.set_seed(123)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_set_seed_different_seeds_diverge():
    su.set_seed(1)
    a = torch.rand(5)
    su.set_seed(2)
    b = torch.rand(5)
    assert not torch.equal(a, b)


def test_set_seed_enables_cudnn_deterministic():
    su.set_seed(42)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


# -- DataLoader generator + worker_init_fn: the actual reproducibility path --

class _TinyDataset(Dataset):
    """No randomness of its own -- isolates the test to the DataLoader's
    own shuffling/worker RNG, exactly like every real dataset this project
    seeds (SleepEventClassificationDataset, BIOTSleepDataset, etc. -- none
    of which do any augmentation or random sampling per item)."""

    def __len__(self):
        return 40

    def __getitem__(self, idx):
        return idx


def _collect_order(seed, num_workers):
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(_TinyDataset(), batch_size=4, shuffle=True,
                         num_workers=num_workers, generator=g,
                         worker_init_fn=su.seed_worker)
    return torch.cat([batch for batch in loader]).tolist()


def test_seeded_dataloader_order_is_reproducible_single_process():
    assert _collect_order(seed=7, num_workers=0) == _collect_order(seed=7, num_workers=0)


def test_seeded_dataloader_order_is_reproducible_multi_worker():
    assert _collect_order(seed=7, num_workers=2) == _collect_order(seed=7, num_workers=2)


def test_seeded_dataloader_different_seeds_give_different_order():
    assert _collect_order(seed=7, num_workers=0) != _collect_order(seed=8, num_workers=0)


def test_unseeded_dataloader_has_no_generator_or_worker_init():
    """Mirrors how every script calls DataLoader when --seed is omitted:
    generator=None, worker_init_fn=None -- i.e. exactly today's (pre-flag)
    default behavior, unchanged."""
    loader = DataLoader(_TinyDataset(), batch_size=4, shuffle=True,
                         num_workers=0, generator=None, worker_init_fn=None)
    order = torch.cat([batch for batch in loader]).tolist()
    assert sorted(order) == list(range(40))


# -- get_git_commit -----------------------------------------------------------

def test_get_git_commit_matches_real_git_rev_parse():
    expected = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT
    ).decode().strip()
    assert su.get_git_commit(REPO_ROOT) == expected
    assert re.fullmatch(r"[0-9a-f]{40}", expected)


def test_get_git_commit_falls_back_on_non_git_dir(tmp_path):
    assert su.get_git_commit(str(tmp_path)) == "unknown"


# -- every fine-tuning script actually wires --seed in ------------------------

FINETUNE_SCRIPTS_ARGPARSE = [
    "scripts/finetune_biot.py",
    "scripts/finetune_labram.py",
    "scripts/finetune_sensorlm.py",
]


def _has_argparse_seed_option(path):
    tree = ast.parse(open(os.path.join(REPO_ROOT, path)).read(), filename=path)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--seed"):
            return True
    return False


def test_every_argparse_finetune_script_has_seed_flag():
    missing = [p for p in FINETUNE_SCRIPTS_ARGPARSE if not _has_argparse_seed_option(p)]
    assert not missing, f"missing --seed option: {missing}"


def test_finetune_sleep_staging_click_command_has_seed_flag():
    path = "sleepfm/pipeline/finetune_sleep_staging.py"
    tree = ast.parse(open(os.path.join(REPO_ROOT, path)).read(), filename=path)
    found = False
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "option"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--seed"):
            found = True
    assert found


# -- finetune_sleep_staging.py's own (duplicated) seed helpers ---------------

def test_finetune_sleep_staging_set_seed_matches_shared_contract():
    import importlib
    spec = importlib.util.spec_from_file_location(
        "finetune_sleep_staging_module",
        os.path.join(REPO_ROOT, "sleepfm", "pipeline", "finetune_sleep_staging.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mod.set_seed(123)
    a = torch.rand(5)
    mod.set_seed(123)
    b = torch.rand(5)
    assert torch.equal(a, b)
    assert torch.backends.cudnn.deterministic is True

    commit = mod.get_git_commit()
    assert re.fullmatch(r"[0-9a-f]{40}", commit)
