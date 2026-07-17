#!/usr/bin/env python3
"""
Diagnostic: sanity-check k-means token viability for next-window prediction.

Loads spectral_cache_pretrain.npz, fits MiniBatchKMeans, then reports:
  - Marginal token distribution and entropy
  - Majority-token-only cross-entropy baseline
  - Self-transition rate (consecutive windows in same file)
  - Copy-previous cross-entropy baseline

Run on a CPU compute node (not login node — numpy hangs there):
    python scripts/diagnose_token_viability.py
    python scripts/diagnose_token_viability.py --n_clusters 256
"""

import argparse
import os
import sys
import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans

# 640 samples = 5 seconds at 128 Hz — used to identify consecutive windows
SPECTRAL_WINDOW = 640


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose k-means token viability for next-window prediction"
    )
    parser.add_argument(
        "--cache_path",
        default="/scratch/project_2019517/sleepfm-data/spectral_cache_pretrain.npz",
        help="Path to spectral_cache_pretrain.npz (default: scratch cache)",
    )
    parser.add_argument(
        "--n_clusters", type=int, default=512,
        help="Number of k-means clusters / vocab size (default 512)",
    )
    parser.add_argument(
        "--km_batch_size", type=int, default=10000,
        help="MiniBatchKMeans partial_fit batch size (default 10000)",
    )
    parser.add_argument(
        "--output_dir", default="diagnostics",
        help="Directory for the text report (default: diagnostics/)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "token_viability_report.txt")

    output_lines = []

    def log(msg=""):
        """Print and buffer for report file."""
        print(msg)
        output_lines.append(str(msg))

    log("=" * 64)
    log("TOKEN VIABILITY DIAGNOSTIC — next-window prediction")
    log("=" * 64)

    # ── 1. Load cache ─────────────────────────────────────────────────────────
    log(f"\n[Step 1] Loading cache: {args.cache_path}")
    if not os.path.isfile(args.cache_path):
        log(f"ERROR: file not found: {args.cache_path}")
        log("Run scripts/run_build_spectral_cache.slurm first.")
        sys.exit(1)

    t0 = time.time()
    # allow_pickle=False is not used because file_paths is a byte-string numpy
    # array that numpy's own format handles, but some numpy versions flag it.
    # We trust this file (our own output from build_spectral_cache.py).
    cache = np.load(args.cache_path, allow_pickle=False)
    log(f"Cache keys: {list(cache.keys())}")
    for k in cache.keys():
        v = cache[k]
        if hasattr(v, "shape") and v.ndim > 0:
            log(f"  {k}: shape={v.shape}  dtype={v.dtype}")
        else:
            log(f"  {k}: {v}")
    log(f"Load time: {time.time() - t0:.1f}s")

    targets = cache["targets"]          # [N, MAX_CH, 5] float32
    masks   = cache["masks"]            # [N, MAX_CH]   bool, True = padded/invalid
    N       = targets.shape[0]
    MAX_CH  = targets.shape[1]
    log(f"\n  N windows : {N:,}")
    log(f"  MAX_CH    : {MAX_CH}")

    has_index = ("file_paths" in cache) and ("window_starts" in cache)
    if has_index:
        file_paths_raw = cache["file_paths"]     # [N] byte strings
        window_starts  = cache["window_starts"]  # [N] int64
        log(f"  Scan index present: file_paths + window_starts ({len(file_paths_raw):,} entries)")
    else:
        log(
            "  Scan index (file_paths / window_starts) NOT present in this cache.\n"
            "  Self-transition analysis will be SKIPPED.\n"
            "  Cause: cache was built before build_spectral_cache.py was updated to\n"
            "  save the scan index.  Rebuild with the updated script to enable it."
        )

    # ── 2. Per-window feature vector ──────────────────────────────────────────
    log(f"\n[Step 2] Building per-window feature vectors")
    log("  Aggregation: mean over valid (unmasked) channels → [N, 5] float32")
    log("  Rationale: spectral bands are comparable across EEG/EOG channels;\n"
        "  mean pooling handles variable channel counts without zero-padding bias.")

    t0 = time.time()
    valid = (~masks).astype(np.float32)          # [N, MAX_CH], 1.0 = real channel
    n_valid_per_win = valid.sum(axis=1)          # [N]
    fully_masked = (n_valid_per_win == 0)
    n_bad = int(fully_masked.sum())
    if n_bad:
        log(f"  WARNING: {n_bad} windows have no valid channels — excluded from k-means")

    # sum(targets * valid) / n_valid  (avoid div-by-zero for fully-masked rows)
    denom = n_valid_per_win.clip(min=1.0)[:, np.newaxis]
    features = (targets * valid[:, :, np.newaxis]).sum(axis=1) / denom  # [N, 5]

    good_mask = ~fully_masked
    features_km = features[good_mask]   # only valid windows go into k-means
    good_idx    = np.where(good_mask)[0]
    n_km = len(features_km)

    log(f"  Feature matrix : {features_km.shape}  dtype={features_km.dtype}")
    log(f"  Feature stats  : mean={features_km.mean():.4f}  std={features_km.std():.4f}"
        f"  min={features_km.min():.4f}  max={features_km.max():.4f}")
    log(f"  Build time: {time.time() - t0:.1f}s")

    # ── 3. MiniBatchKMeans ────────────────────────────────────────────────────
    log(f"\n[Step 3] MiniBatchKMeans  n_clusters={args.n_clusters}  "
        f"batch_size={args.km_batch_size}")
    log(f"  {n_km:,} windows  →  ~{n_km // args.km_batch_size + 1} batches total")

    km = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        random_state=42,
        batch_size=args.km_batch_size,
        max_iter=100,
        n_init=3,
        reassignment_ratio=0.01,
        verbose=0,
    )

    t0 = time.time()
    batch_count = 0
    for start in range(0, n_km, args.km_batch_size):
        km.partial_fit(features_km[start : start + args.km_batch_size])
        batch_count += 1
        if batch_count == 1 or batch_count % 50 == 0:
            done_pct = 100.0 * min(start + args.km_batch_size, n_km) / n_km
            log(f"  partial_fit batch {batch_count:4d} | {done_pct:5.1f}% | {time.time()-t0:.0f}s elapsed")

    log(f"  KMeans fit complete in {time.time() - t0:.1f}s")

    # ── 4. Token assignment ───────────────────────────────────────────────────
    log(f"\n[Step 4] Assigning tokens to {n_km:,} windows ...")
    t0 = time.time()
    labels_km = np.empty(n_km, dtype=np.int32)
    for start in range(0, n_km, args.km_batch_size):
        labels_km[start : start + args.km_batch_size] = km.predict(
            features_km[start : start + args.km_batch_size]
        )

    # Full labels array (fully-masked windows get sentinel -1)
    labels = np.full(N, -1, dtype=np.int32)
    labels[good_idx] = labels_km
    log(f"  Assignment complete in {time.time() - t0:.1f}s")

    # ── 5a. Marginal distribution ─────────────────────────────────────────────
    log(f"\n[Step 5a] Marginal token distribution")
    counts = np.bincount(labels_km, minlength=args.n_clusters).astype(np.int64)
    probs  = counts / n_km
    sort_idx = np.argsort(counts)[::-1]

    log(f"\n  Top 10 tokens (most frequent) — out of {args.n_clusters}:")
    log(f"  {'Rank':>5}  {'Token':>7}  {'Count':>12}  {'%':>8}")
    for rank, tid in enumerate(sort_idx[:10]):
        log(f"  {rank+1:>5}  {tid:>7}  {counts[tid]:>12,}  {100*probs[tid]:>7.3f}%")

    log(f"\n  Bottom 10 tokens (least frequent):")
    log(f"  {'Rank':>5}  {'Token':>7}  {'Count':>12}  {'%':>8}")
    for rank, tid in enumerate(reversed(sort_idx[-10:])):
        log(f"  {args.n_clusters-rank:>5}  {tid:>7}  {counts[tid]:>12,}  {100*probs[tid]:>7.3f}%")

    # ── 5b. Entropy ───────────────────────────────────────────────────────────
    log(f"\n[Step 5b] Entropy of marginal distribution")
    nonzero_p = probs[probs > 0]
    entropy_bits = float(-np.sum(nonzero_p * np.log2(nonzero_p)))
    max_entropy  = float(np.log2(args.n_clusters))
    n_empty = int((counts == 0).sum())
    log(f"  Marginal entropy : {entropy_bits:.4f} / {max_entropy:.2f} bits")
    log(f"  Relative entropy : {100*entropy_bits/max_entropy:.1f}% of maximum")
    if n_empty:
        log(f"  Empty clusters   : {n_empty} / {args.n_clusters} tokens never assigned")
    else:
        log(f"  Empty clusters   : 0 (all {args.n_clusters} tokens used)")

    # ── 5c. Majority-token baseline CE ────────────────────────────────────────
    log(f"\n[Step 5c] Majority-token baseline cross-entropy  (nats, natural log)")
    p_max      = float(probs.max())
    majority_ce = float(-np.log(p_max))
    random_ce  = float(np.log(args.n_clusters))
    log(f"  Most common token : #{sort_idx[0]}  p={p_max:.6f}")
    log(f"  CE (always predict most common) : {majority_ce:.4f} nats")
    log(f"  CE (random uniform guess)       : {random_ce:.4f} nats  [= log({args.n_clusters})]")

    # ── 5d–e. Self-transition & copy-previous CE ──────────────────────────────
    self_trans_rate = None
    copy_prev_ce    = None

    log(f"\n[Step 5d] Self-transition rate")
    if not has_index:
        log("  SKIPPED — file_paths/window_starts not saved in this cache.")
        log("  Rebuild the spectral cache with the updated build_spectral_cache.py")
        log("  (the version that saves file_paths and window_starts to the .npz)")
        log("  then rerun this script.")
    else:
        fps = np.array(
            [p.decode() if isinstance(p, bytes) else str(p) for p in file_paths_raw]
        )
        ws = window_starts.astype(np.int64)

        # A consecutive pair (i, i+1) is valid iff same file AND
        # window_starts differ by exactly SPECTRAL_WINDOW (= 640 samples).
        same_file   = fps[:-1] == fps[1:]
        consec_step = ws[1:] == ws[:-1] + SPECTRAL_WINDOW
        pair_mask   = same_file & consec_step

        n_pairs = int(pair_mask.sum())
        log(f"  Consecutive within-file pairs : {n_pairs:,}")

        if n_pairs == 0:
            log("  No consecutive pairs found — check that window_starts are correct.")
        else:
            tok_t  = labels[:-1][pair_mask]
            tok_t1 = labels[1:][pair_mask]

            # Exclude pairs where either window was fully masked (label == -1)
            valid_pairs = (tok_t >= 0) & (tok_t1 >= 0)
            n_valid_pairs = int(valid_pairs.sum())
            log(f"  Valid pairs (both windows labeled) : {n_valid_pairs:,}")

            if n_valid_pairs == 0:
                log("  No valid pairs — self-transition analysis skipped.")
            else:
                tok_t  = tok_t[valid_pairs]
                tok_t1 = tok_t1[valid_pairs]

                same_tok = tok_t == tok_t1
                self_trans_rate = float(same_tok.mean())
                log(f"  Self-transition rate : {100*self_trans_rate:.2f}%  "
                    f"({same_tok.sum():,} / {n_valid_pairs:,} pairs)")

                # Per-token self-transition breakdown
                diag_counts  = np.bincount(tok_t[same_tok],  minlength=args.n_clusters)
                tok_t_counts = np.bincount(tok_t, minlength=args.n_clusters)
                with np.errstate(divide="ignore", invalid="ignore"):
                    per_tok_self = np.where(
                        tok_t_counts > 0,
                        diag_counts / tok_t_counts.astype(float),
                        0.0,
                    )
                log(f"\n  Per-token self-transition rate stats:")
                log(f"    mean   = {per_tok_self.mean():.4f}")
                log(f"    median = {np.median(per_tok_self):.4f}")
                log(f"    min    = {per_tok_self.min():.4f}")
                log(f"    max    = {per_tok_self.max():.4f}")

                log(f"\n[Step 5e] Copy-previous baseline cross-entropy  (nats)")
                # Empirical CE of always predicting token(t) as the label for token(t+1).
                # A "copy-previous" policy is a deterministic one-hot on token(t), so its
                # CE against the true label is 0 when correct and +∞ when wrong.
                # The spec asks for -log(P_same) where P_same = self_trans_rate, treating
                # the copy policy as if it assigned probability self_trans_rate to "same".
                copy_prev_ce = float(-np.log(self_trans_rate)) if self_trans_rate > 0 else float("inf")
                log(f"  CE (copy-previous) = -log(self_trans_rate)")
                log(f"                     = -log({self_trans_rate:.6f})")
                log(f"                     = {copy_prev_ce:.4f} nats")
                log(f"  Interpretation: a model that simply copies the previous token")
                log(f"  achieves this CE as a lower-bound estimate.  A well-trained")
                log(f"  predictor must beat this number meaningfully to be non-trivial.")

    # ── Summary ───────────────────────────────────────────────────────────────
    log("")
    log("================================")
    log("=== TOKEN VIABILITY SUMMARY ===")
    log(f"N windows: {N:,}")
    log(f"N clusters: {args.n_clusters}")
    log(f"Marginal entropy: {entropy_bits:.2f} / {max_entropy:.2f} bits")
    log(f"Majority-token baseline CE: {majority_ce:.4f}")
    if self_trans_rate is not None:
        log(f"Self-transition rate: {100*self_trans_rate:.1f}%")
        log(f"Copy-previous baseline CE: {copy_prev_ce:.4f}")
    else:
        log("Self-transition rate: N/A (index not in cache — see Step 5d above)")
        log("Copy-previous baseline CE: N/A")
    log(f"(For comparison: random-guess CE = log({args.n_clusters}) = {random_ce:.2f})")
    log("================================")

    # ── Save report ───────────────────────────────────────────────────────────
    with open(report_path, "w") as f:
        f.write("\n".join(output_lines) + "\n")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
