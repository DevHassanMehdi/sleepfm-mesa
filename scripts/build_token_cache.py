#!/usr/bin/env python3
"""
Build k-means token cache for next-window prediction pretraining (Stage 0).

Loads spectral_cache_pretrain.npz and spectral_cache_validation.npz, fits
MiniBatchKMeans on pretrain features only, assigns token indices to both
splits, and writes consecutive within-file pair arrays for training.

Run on a CPU interactive/small node (NOT the login node):
    sinteractive --account=project_2019517 --partition=small \\
                 -c 4 --mem=32G --time=01:00:00
    source /scratch/project_2019517/miniconda3/etc/profile.d/conda.sh
    conda activate sleepfm_env
    cd /users/hamehdi/projects/sleepfm-mesa
    python scripts/build_token_cache.py
"""

import argparse
import os
import sys
import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans

SPECTRAL_WINDOW = 640   # 5 seconds at 128 Hz — must match build_spectral_cache.py
N_CLUSTERS = 512
KM_BATCH_SIZE = 10000
CACHE_DIR = "/scratch/project_2019517/sleepfm-data"
REPORT_DIR = "diagnostics"


def load_and_validate_cache(path, name):
    """Load npz cache; raise loudly if required keys are missing or counts mismatch."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{name} cache not found: {path}")

    cache = np.load(path, allow_pickle=False)
    print(f"  [{name}] keys: {list(cache.keys())}")

    required = ("targets", "masks", "n_windows", "file_paths", "window_starts")
    for key in required:
        if key not in cache:
            raise KeyError(
                f"Required key '{key}' missing from {name} cache.\n"
                f"Path: {path}\n"
                f"Present keys: {list(cache.keys())}\n"
                "Rebuild the spectral cache with the updated build_spectral_cache.py."
            )

    targets = cache["targets"]                           # [N, MAX_CH, 5] float32
    masks = cache["masks"]                               # [N, MAX_CH] bool, True=padded
    n_windows = int(cache["n_windows"])
    file_paths_raw = cache["file_paths"]                 # [N] byte strings
    window_starts = cache["window_starts"].astype(np.int64)

    if targets.shape[0] != n_windows:
        raise ValueError(
            f"{name}: targets.shape[0]={targets.shape[0]} != n_windows={n_windows}. "
            "Cache file may be corrupted."
        )
    if len(file_paths_raw) != n_windows:
        raise ValueError(
            f"{name}: len(file_paths)={len(file_paths_raw)} != n_windows={n_windows}."
        )
    if len(window_starts) != n_windows:
        raise ValueError(
            f"{name}: len(window_starts)={len(window_starts)} != n_windows={n_windows}."
        )

    fps = np.array(
        [p.decode() if isinstance(p, bytes) else str(p) for p in file_paths_raw]
    )
    return targets, masks, fps, window_starts, n_windows


def build_features(targets, masks):
    """
    Mean over valid (unmasked) channels → [N, 5] float32.
    Exactly matches the aggregation in diagnose_token_viability.py.
    Fully-masked windows (no valid channels) are flagged in the returned bool array.
    """
    valid = (~masks).astype(np.float32)          # [N, MAX_CH], 1.0 = real channel
    n_valid_per_win = valid.sum(axis=1)          # [N]
    fully_masked = n_valid_per_win == 0
    denom = n_valid_per_win.clip(min=1.0)[:, np.newaxis]
    features = (targets * valid[:, :, np.newaxis]).sum(axis=1) / denom  # [N, 5]
    return features, fully_masked


def assign_tokens_batched(km, features, fully_masked, n_windows):
    """
    Assign token indices to all windows.
    Fully-masked windows receive sentinel label -1 and are excluded from km.predict().
    """
    good_mask = ~fully_masked
    good_idx = np.where(good_mask)[0]
    features_km = features[good_idx]
    n_km = len(features_km)

    labels_km = np.empty(n_km, dtype=np.int32)
    for start in range(0, n_km, KM_BATCH_SIZE):
        labels_km[start:start + KM_BATCH_SIZE] = km.predict(
            features_km[start:start + KM_BATCH_SIZE]
        )

    labels = np.full(n_windows, -1, dtype=np.int32)
    labels[good_idx] = labels_km
    return labels


def build_pairs(fps, window_starts, labels):
    """
    For each window i, check if (fps[i], window_starts[i] + SPECTRAL_WINDOW) exists
    in the same split's cache. If yes, and both windows have label >= 0, record
    (i, labels[j]) as a valid training pair.

    Returns int64 array of shape [M, 2]: column 0 = current window index,
    column 1 = next window's token label.
    """
    lookup = {}
    for i in range(len(fps)):
        lookup[(fps[i], int(window_starts[i]))] = i

    pairs = []
    for i in range(len(fps)):
        if labels[i] < 0:
            continue
        next_key = (fps[i], int(window_starts[i]) + SPECTRAL_WINDOW)
        j = lookup.get(next_key)
        if j is None:
            continue
        if labels[j] < 0:
            continue
        pairs.append((i, int(labels[j])))

    return (
        np.array(pairs, dtype=np.int64)
        if pairs
        else np.empty((0, 2), dtype=np.int64)
    )


def entropy_bits(labels, n_clusters):
    valid = labels[labels >= 0]
    counts = np.bincount(valid.astype(np.int64), minlength=n_clusters)
    probs = counts / counts.sum()
    nz = probs[probs > 0]
    return float(-np.sum(nz * np.log2(nz))), counts, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default=CACHE_DIR,
                         help="Directory holding spectral_cache_{pretrain,validation}.npz "
                              "and where token_codebook.npy etc. will be written. "
                              "Default is the 350-subject Puhti-era location.")
    parser.add_argument("--report_name", default="token_cache_report.txt",
                         help="Filename (under diagnostics/) for the build report. "
                              "Change this when pointing at a different cache_dir so "
                              "the historical report isn't overwritten.")
    args = parser.parse_args()

    cache_dir = args.cache_dir
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, args.report_name)

    lines = []

    def log(msg=""):
        print(msg)
        lines.append(str(msg))

    t_total = time.time()

    log("=" * 70)
    log("TOKEN CACHE BUILD — k-means codebook for next-window prediction")
    log(f"n_clusters={N_CLUSTERS}  batch_size={KM_BATCH_SIZE}  "
        f"SPECTRAL_WINDOW={SPECTRAL_WINDOW}")
    log("=" * 70)

    # ── 1. Load both caches ───────────────────────────────────────────────────
    log(f"\n[Step 1] Loading spectral caches from {cache_dir}/")

    pretrain_path = os.path.join(cache_dir, "spectral_cache_pretrain.npz")
    val_path = os.path.join(cache_dir, "spectral_cache_validation.npz")

    t0 = time.time()
    targets_pre, masks_pre, fps_pre, ws_pre, N_pre = load_and_validate_cache(
        pretrain_path, "pretrain"
    )
    targets_val, masks_val, fps_val, ws_val, N_val = load_and_validate_cache(
        val_path, "validation"
    )
    log(f"  Pretrain  : {N_pre:,} windows  targets={targets_pre.shape}")
    log(f"  Validation: {N_val:,} windows  targets={targets_val.shape}")
    log(f"  Load time : {time.time() - t0:.1f}s")

    # ── 2. Feature vectors ────────────────────────────────────────────────────
    log(f"\n[Step 2] Building per-window feature vectors [N, 5]")
    log("  Aggregation: mean over valid (unmasked) channels — matches diagnostic script")

    t0 = time.time()
    features_pre, masked_pre = build_features(targets_pre, masks_pre)
    features_val, masked_val = build_features(targets_val, masks_val)

    n_bad_pre = int(masked_pre.sum())
    n_bad_val = int(masked_val.sum())
    if n_bad_pre:
        log(f"  WARNING: {n_bad_pre} pretrain windows fully masked (will get label=-1)")
    if n_bad_val:
        log(f"  WARNING: {n_bad_val} validation windows fully masked (will get label=-1)")

    features_pre_km = features_pre[~masked_pre]
    log(f"  Pretrain k-means features : {features_pre_km.shape}  dtype={features_pre_km.dtype}")
    log(
        f"  Feature stats: mean={features_pre_km.mean():.4f}  "
        f"std={features_pre_km.std():.4f}  "
        f"min={features_pre_km.min():.4f}  max={features_pre_km.max():.4f}"
    )
    log(f"  Build time: {time.time() - t0:.1f}s")

    # ── 3. Fit MiniBatchKMeans on pretrain only ───────────────────────────────
    log(f"\n[Step 3] Fitting MiniBatchKMeans on pretrain split ONLY")
    log("  (Validation is withheld to avoid leakage — assigned only)")

    km = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        batch_size=KM_BATCH_SIZE,
        max_iter=100,
        n_init=3,
        reassignment_ratio=0.01,
        verbose=0,
    )

    n_km = len(features_pre_km)
    t0 = time.time()
    batch_count = 0
    for start in range(0, n_km, KM_BATCH_SIZE):
        km.partial_fit(features_pre_km[start:start + KM_BATCH_SIZE])
        batch_count += 1
        if batch_count == 1 or batch_count % 50 == 0:
            done_pct = 100.0 * min(start + KM_BATCH_SIZE, n_km) / n_km
            log(f"  partial_fit batch {batch_count:4d} | {done_pct:5.1f}% | "
                f"{time.time()-t0:.0f}s elapsed")

    log(f"  KMeans fit complete in {time.time() - t0:.1f}s")

    # ── 4. Assign tokens ──────────────────────────────────────────────────────
    log(f"\n[Step 4] Assigning tokens")

    t0 = time.time()
    labels_pre = assign_tokens_batched(km, features_pre, masked_pre, N_pre)
    log(f"  Pretrain  : {(labels_pre >= 0).sum():,} / {N_pre:,} windows assigned")
    labels_val = assign_tokens_batched(km, features_val, masked_val, N_val)
    log(f"  Validation: {(labels_val >= 0).sum():,} / {N_val:,} windows assigned")
    log(f"  Assignment time: {time.time() - t0:.1f}s")

    # Check for empty clusters — fail loudly as specified
    valid_labels_pre = labels_pre[labels_pre >= 0]
    cluster_counts = np.bincount(valid_labels_pre.astype(np.int64), minlength=N_CLUSTERS)
    n_empty = int((cluster_counts == 0).sum())
    if n_empty > 0:
        raise RuntimeError(
            f"Degenerate codebook: {n_empty}/{N_CLUSTERS} clusters have 0 assigned "
            "pretrain windows. Consider reducing n_clusters or increasing data."
        )
    log(f"  Empty cluster check: PASSED — all {N_CLUSTERS} clusters populated")

    # ── 5. Build consecutive pairs ────────────────────────────────────────────
    log(f"\n[Step 5] Building consecutive within-file pairs")
    log(f"  Definition: (fps[i], ws[i]+{SPECTRAL_WINDOW}) must exist in same split's cache")
    log("  Both the current and next window must have label >= 0")

    t0 = time.time()
    pairs_pre = build_pairs(fps_pre, ws_pre, labels_pre)
    pairs_val = build_pairs(fps_val, ws_val, labels_val)

    pct_excl_pre = 100.0 * (1 - len(pairs_pre) / max(N_pre, 1))
    pct_excl_val = 100.0 * (1 - len(pairs_val) / max(N_val, 1))
    log(f"  Pretrain  : {len(pairs_pre):,} valid pairs  "
        f"({pct_excl_pre:.1f}% excluded by file boundaries / masking)")
    log(f"  Validation: {len(pairs_val):,} valid pairs  "
        f"({pct_excl_val:.1f}% excluded)")
    log(f"  Pair build time: {time.time() - t0:.1f}s")

    # ── 6. Save outputs ───────────────────────────────────────────────────────
    log(f"\n[Step 6] Saving outputs to {cache_dir}/")

    t0 = time.time()
    paths = {
        "token_codebook.npy":               km.cluster_centers_.astype(np.float32),
        "token_assignments_pretrain.npy":   labels_pre,
        "token_assignments_validation.npy": labels_val,
        "token_pairs_pretrain.npy":         pairs_pre,
        "token_pairs_validation.npy":       pairs_val,
    }
    for fname, arr in paths.items():
        fpath = os.path.join(cache_dir, fname)
        np.save(fpath, arr)
        log(f"  {fname:<40} shape={arr.shape}  dtype={arr.dtype}")
    log(f"  Save time: {time.time() - t0:.1f}s")

    # ── 7. Distribution report ────────────────────────────────────────────────
    log(f"\n[Step 7] Token distribution and entropy")

    max_ent = float(np.log2(N_CLUSTERS))
    ent_pre, cnt_pre, prob_pre = entropy_bits(labels_pre, N_CLUSTERS)
    ent_val, cnt_val, prob_val = entropy_bits(labels_val, N_CLUSTERS)

    log(f"\n  Pretrain   entropy: {ent_pre:.4f} / {max_ent:.2f} bits "
        f"({100*ent_pre/max_ent:.1f}% of max)")
    log(f"  Validation entropy: {ent_val:.4f} / {max_ent:.2f} bits "
        f"({100*ent_val/max_ent:.1f}% of max)")

    ent_diff = abs(ent_pre - ent_val)
    if ent_diff > 0.5:
        log(f"\n  !! FLAG: Entropy gap = {ent_diff:.4f} bits (threshold 0.5)")
        log("  !! Validation distribution differs substantially from pretrain.")
        log("  !! The codebook may not generalize well — inspect per-cluster counts.")
    else:
        log(f"\n  Entropy gap (|val - pre|): {ent_diff:.4f} bits — distributions consistent.")

    n_empty_val = int((cnt_val == 0).sum())
    if n_empty_val > 0:
        log(f"  NOTE: {n_empty_val} clusters unseen in validation (normal for smaller split).")

    # Self-transition rate
    log(f"\n  Self-transition rate:")

    def self_trans(pairs, labels_arr, split):
        if len(pairs) == 0:
            log(f"    {split}: N/A (no valid pairs)")
            return None
        cur_tok = labels_arr[pairs[:, 0]]
        nxt_tok = pairs[:, 1]
        same = cur_tok == nxt_tok
        rate = float(same.mean())
        log(f"    {split}: {100*rate:.2f}%  ({same.sum():,} / {len(pairs):,} pairs)")
        return rate

    rate_pre = self_trans(pairs_pre, labels_pre, "pretrain  ")
    rate_val = self_trans(pairs_val, labels_val, "validation")

    if rate_pre is not None:
        diff_pp = abs(rate_pre - 0.1704) * 100
        if diff_pp <= 1.0:
            log(f"\n    CONFIRMED: matches diagnostic's 17.04% within 1 pp "
                f"(diff={diff_pp:.2f} pp)")
        else:
            log(f"\n    NOTE: diagnostic reported 17.04%; observed {100*rate_pre:.2f}% "
                f"(diff={diff_pp:.2f} pp)")
            log("    Possible causes: different random seed initialisation path,")
            log("    dict-lookup vs adjacent-array pair construction, or stochastic")
            log("    MiniBatchKMeans convergence variation between runs.")

    # ── Summary ───────────────────────────────────────────────────────────────
    t_elapsed = time.time() - t_total
    log("")
    log("=" * 70)
    log("=== TOKEN CACHE BUILD SUMMARY ===")
    log(f"  Pretrain  : {N_pre:,} windows | {len(pairs_pre):,} valid pairs "
        f"({pct_excl_pre:.1f}% excluded)")
    log(f"  Validation: {N_val:,} windows | {len(pairs_val):,} valid pairs "
        f"({pct_excl_val:.1f}% excluded)")
    log(f"  Codebook  : {N_CLUSTERS} clusters, 0 empty (checked on pretrain)")
    log(f"  Entropy   : pretrain={ent_pre:.2f} bits, validation={ent_val:.2f} bits "
        f"(max={max_ent:.2f})")
    if rate_pre is not None:
        log(f"  Self-trans: pretrain={100*rate_pre:.2f}% (diagnostic ref: 17.04%)")
    log(f"  Total time: {t_elapsed:.1f}s")
    log("=" * 70)
    log(f"\nReport: {report_path}")

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
