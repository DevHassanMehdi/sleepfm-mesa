"""Regression tests for the label-validity/structural-mask fix in
SleepEventLSTMClassifier.forward() (see commit 1da1df7). Synthetic data
only -- no real checkpoints, no real MESA/Moody files, no GPU, no
cluster-specific paths, so this runs anywhere. Plain assertions (no
pytest dependency, matching the rest of this repo).
"""
import os
import sys

import torch

SLEEPFM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sleepfm")
sys.path.insert(0, SLEEPFM_DIR)

from models.dataset import sleep_event_finetune_full_collate_fn  # noqa: E402
from models.models import SleepEventLSTMClassifier  # noqa: E402


MODEL_PARAMS = dict(embed_dim=16, num_heads=2, num_layers=1, num_classes=5,
                     pooling_head=2, dropout=0.0, max_seq_length=64)


def make_synthetic_item(seed, n_channels=1, n_epochs=40, embed_dim=16,
                         max_channels=4, unscored_indices=()):
    """One collate_fn-shaped item: (x, y, max_channels, max_seq_len,
    hdf5_path, label_valid). x/y are deterministic given `seed`, regardless
    of unscored_indices, so two calls with the same seed but different
    unscored_indices produce identical data and differ only in which
    epochs are marked unscored."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_channels, n_epochs, embed_dim, generator=g)
    y = torch.randint(0, 5, (n_epochs,), generator=g).float()
    label_valid = torch.ones(n_epochs, dtype=torch.bool)
    for idx in unscored_indices:
        label_valid[idx] = False
    return x, y, max_channels, None, "synthetic.hdf5", label_valid


def build_model(seed):
    torch.manual_seed(seed)
    model = SleepEventLSTMClassifier(**MODEL_PARAMS)
    model.eval()
    return model


def test_forward_pass_invariant_to_label_validity():
    """Test B: logits must not change when ONLY label validity changes,
    as long as the same structural_mask (padding only) is used both
    times -- label validity must affect the loss/metrics, never the
    forward pass."""
    model = build_model(seed=1)
    item_real = make_synthetic_item(seed=2, unscored_indices=(0, 5, 17, 30))
    item_allvalid = make_synthetic_item(seed=2, unscored_indices=())

    x_r, y_r, mask_r, struct_r, _ = sleep_event_finetune_full_collate_fn([item_real])
    x_a, y_a, mask_a, struct_a, _ = sleep_event_finetune_full_collate_fn([item_allvalid])

    assert torch.equal(x_r, x_a), "synthetic x_data must be identical between the two scenarios"
    assert torch.equal(struct_r, struct_a), "structural_mask must never depend on label validity"
    assert not torch.equal(mask_r, mask_a), "combined mask SHOULD differ -- that's the point of this test"

    with torch.no_grad():
        logits_real, _ = model(x_r, mask_r, struct_r)
        logits_allvalid, _ = model(x_a, mask_a, struct_a)

    assert torch.equal(logits_real, logits_allvalid), (
        "forward-pass logits must be EXACTLY invariant to label validity "
        "when structural_mask is supplied"
    )
    print("PASS: test_forward_pass_invariant_to_label_validity")


def test_mesa_regression_no_unscored_epochs():
    """'MESA regression', synthetic form: MESA never has unscored (-1)
    epochs, so structural_mask and the combined mask coincide exactly for
    every real MESA file. Confirms the fixed code path (structural_mask
    passed) agrees exactly with the pre-fix code path (structural_mask
    omitted, old mask[:, :, 0] sampling) whenever there are no unscored
    epochs -- i.e. the fix is a no-op for MESA."""
    model = build_model(seed=3)
    item = make_synthetic_item(seed=4, unscored_indices=())  # no unscored epochs at all, like MESA

    x, y, mask, struct, _ = sleep_event_finetune_full_collate_fn([item])
    assert torch.equal(mask, struct), (
        "with no unscored epochs, the combined mask and the structural "
        "mask must be identical -- this is exactly MESA's situation"
    )

    with torch.no_grad():
        logits_fixed, _ = model(x, mask, struct)   # new code path
        logits_fallback, _ = model(x, mask)        # pre-fix code path (structural_mask omitted)

    assert torch.equal(logits_fixed, logits_fallback), (
        "when there are no unscored epochs, the fixed and pre-fix code "
        "paths must agree exactly -- MESA (which never has -1 labels) "
        "must be unaffected by this fix"
    )
    print("PASS: test_mesa_regression_no_unscored_epochs")


def test_no_nan_when_first_epoch_unscored():
    """The actual bug this fix addresses: if epoch 0 is unscored (as on
    nearly every Moody record), the old mask[:, :, 0] sampling marked the
    real channel as fully padded for the WHOLE file, collapsing spatial
    pooling to an all-masked attention query and producing NaN logits.
    Confirms the fixed path stays finite."""
    model = build_model(seed=5)
    item = make_synthetic_item(seed=6, unscored_indices=(0,))
    x, y, mask, struct, _ = sleep_event_finetune_full_collate_fn([item])

    with torch.no_grad():
        logits, _ = model(x, mask, struct)

    assert not torch.isnan(logits).any(), "fixed forward pass must not produce NaN when epoch 0 is unscored"
    assert not torch.isinf(logits).any(), "fixed forward pass must not produce Inf when epoch 0 is unscored"
    print("PASS: test_no_nan_when_first_epoch_unscored")


def main():
    test_forward_pass_invariant_to_label_validity()
    test_mesa_regression_no_unscored_epochs()
    test_no_nan_when_first_epoch_unscored()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
