"""Next-window token prediction pretraining for SleepFM SetTransformer encoder.

The encoder observes a 5-second (640-sample) multi-channel window and must
predict the discrete token ID of the NEXT consecutive window in the same file.
Token IDs are k-means cluster labels (n_clusters=512) from build_token_cache.py.

Loss: cross-entropy over 512 classes. Cannot collapse to a trivial single-point
solution the way MSE/cosine objectives did (see commits a66fa67, 3915a7d).

Reference baselines (printed at startup):
  - Random guess:    CE = log(512) = 6.2383 nats  [where we start]
  - Copy-previous:   CE = 1.7694 nats             [must beat this to be non-trivial]

Output: checkpoints/encoders/nexttoken/SetTransformer/nexttoken_128_patch_size_640/
"""

import datetime
import math
import os
import sys

import click
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from loguru import logger
from torch import nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.models import SetTransformer
from utils import load_config, save_data

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

WINDOW_SIZE = 640
MAX_CHANNELS = 10
COPY_PREV_CE = 1.7694      # from diagnostics/token_cache_report.txt
COLLAPSE_THRESHOLD = 0.10  # train CE below this is almost certainly collapse


def _has_nan_or_inf(state_dict):
    return any(
        torch.isnan(v).any().item() or torch.isinf(v).any().item()
        for v in state_dict.values()
        if isinstance(v, torch.Tensor)
    )


class TokenHead(nn.Module):
    """Linear(embed_dim → n_clusters) next-token classification head."""

    def __init__(self, embed_dim=128, n_clusters=512):
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_clusters)

    def forward(self, x):
        return self.fc(x)


class NextTokenDataset(torch.utils.data.Dataset):
    """Cache-backed dataset for next-window token prediction.

    Loads without any HDF5 I/O at training time:
      - token_pairs_{split}.npy  — [M, 2] int64: (current_idx, next_token_id)
      - spectral_signals_{split}.bin — [N, MAX_CH, 640] float32 memmap
      - spectral_cache_{split}.npz   — masks [N, MAX_CH] bool (True = padded)

    __getitem__(i) returns (signal [MAX_CH, 640], mask [MAX_CH], next_token int).
    No file-boundary logic needed — pairs are already filtered by build_token_cache.py.
    """

    def __init__(self, config, split="pretrain"):
        cache_dir = config.get("spectral_cache_dir", "/scratch/project_2019517/sleepfm-data")
        max_channels = config.get("max_channels", MAX_CHANNELS)

        # Token pairs: (current_window_idx, next_token_label)
        pairs_path = config.get(f"token_pairs_{split}_path")
        if not pairs_path or not os.path.isfile(pairs_path):
            raise FileNotFoundError(
                f"Token pairs not found for split='{split}'.\n"
                f"  Config key: token_pairs_{split}_path\n"
                f"  Path: {pairs_path}\n"
                f"  Run: python scripts/build_token_cache.py"
            )
        self.pairs = np.load(pairs_path)  # [M, 2] int64

        # Masks from spectral cache (small enough to stay fully in RAM)
        cache_path = os.path.join(cache_dir, f"spectral_cache_{split}.npz")
        if not os.path.isfile(cache_path):
            raise FileNotFoundError(f"Spectral cache not found: {cache_path}")
        cache = np.load(cache_path, allow_pickle=False)
        self.masks = cache["masks"]      # [N, MAX_CH] bool, True = padded
        N = int(cache["n_windows"])

        # Raw signals via memmap — OS pages only requested rows from disk
        signals_path = config.get(
            f"spectral_signals_path_{split}",
            os.path.join(cache_dir, f"spectral_signals_{split}.bin"),
        )
        if not os.path.isfile(signals_path):
            raise FileNotFoundError(f"Signals memmap not found: {signals_path}")
        self.signals = np.memmap(
            signals_path, dtype="float32", mode="r",
            shape=(N, max_channels, WINDOW_SIZE),
        )

        logger.info(
            f"NextTokenDataset [{split}]: {len(self.pairs):,} pairs "
            f"from {N:,} windows"
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        idx = int(self.pairs[i, 0])
        next_token = int(self.pairs[i, 1])
        signal = np.array(self.signals[idx], dtype=np.float32)  # copy — safe for workers
        mask = self.masks[idx].copy()
        return (
            torch.from_numpy(signal),   # [MAX_CH, 640]
            torch.from_numpy(mask),     # [MAX_CH] bool
            next_token,                 # int — collated to [B] LongTensor by DataLoader
        )


def run_epoch(loader, encoder, head, optimizer, device, split, n_clusters, epoch):
    is_train = split == "pretrain"
    encoder.train()  # ALWAYS — keeps BatchNorm in train mode even during val pass
    head.train()

    total_loss = 0.0
    total_n = 0

    with torch.set_grad_enabled(is_train):
        with tqdm.tqdm(total=len(loader), desc=f"[{split}]") as pbar:
            for signals, masks, next_tokens in loader:
                try:
                    signals = signals.to(device)                         # [B, MAX_CH, 640]
                    masks = masks.to(device)                             # [B, MAX_CH] bool
                    next_tokens = next_tokens.to(device, dtype=torch.long)  # [B]

                    emb, _ = encoder(signals, masks)  # [B, embed_dim]
                    logits = head(emb)                # [B, n_clusters]
                    loss = F.cross_entropy(logits, next_tokens)

                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"[{split}] NaN/Inf loss — skipping batch")
                        pbar.update()
                        continue

                    if is_train:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    B = signals.size(0)
                    total_loss += loss.item() * B
                    total_n += B

                except Exception as exc:
                    logger.warning(f"[{split}] batch error ({exc}) — skipping")

                pbar.set_postfix_str(f"CE={total_loss / max(total_n, 1):.4f}")
                pbar.update()

    epoch_ce = total_loss / max(total_n, 1)

    if is_train and epoch_ce < COLLAPSE_THRESHOLD:
        logger.warning(
            f"\n!!! COLLAPSE WARNING: train CE = {epoch_ce:.4f} at epoch {epoch}.\n"
            f"    Threshold: {COLLAPSE_THRESHOLD}  |  "
            f"Random-guess CE: {math.log(n_clusters):.4f}  |  "
            f"Copy-previous CE: {COPY_PREV_CE:.4f}\n"
            f"    Training has likely collapsed to a trivial solution.\n"
            f"    Inspect predicted token distributions before continuing."
        )

    return epoch_ce


@click.command("pretrain_nexttoken")
@click.option("--config_path", default="sleepfm/configs/config_pretrain_nexttoken.yaml")
@click.option("--checkpoint_path", default=None,
              help="Resume from an existing checkpoint directory (overrides save_path).")
def pretrain_nexttoken(config_path, checkpoint_path):
    config = load_config(config_path)

    if checkpoint_path:
        output = checkpoint_path
        config = load_config(os.path.join(output, "config.json"))
    else:
        output = os.path.join(
            config["save_path"],
            f"SetTransformer/nexttoken_{config['embed_dim']}_patch_size_{config['patch_size']}",
        )
        os.makedirs(output, exist_ok=True)

    logger.info(f"Output: {output}")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Device: {device}")

    # ── Codebook: read n_clusters from file, validate against config ──────────
    codebook_path = config["token_codebook_path"]
    if not os.path.isfile(codebook_path):
        raise FileNotFoundError(f"Token codebook not found: {codebook_path}")
    codebook = np.load(codebook_path)
    n_clusters = int(codebook.shape[0])
    n_clusters_cfg = config.get("n_clusters")
    if n_clusters_cfg is not None and n_clusters != int(n_clusters_cfg):
        raise ValueError(
            f"Codebook has {n_clusters} clusters but config says n_clusters={n_clusters_cfg}. "
            f"Update config_pretrain_nexttoken.yaml or rebuild the codebook with the same k."
        )
    logger.info(f"Token codebook: {n_clusters} clusters  feature_dim={codebook.shape[1]}")

    # ── Reference baselines ───────────────────────────────────────────────────
    random_ce = math.log(n_clusters)
    logger.info(
        f"\nReference baselines (cross-entropy, nats):\n"
        f"  Random guess   : {random_ce:.4f}  [= log({n_clusters})]  ← where random-init starts\n"
        f"  Copy-previous  : {COPY_PREV_CE:.4f}  ← must beat this to be non-trivial\n"
        f"  Collapse flag  : < {COLLAPSE_THRESHOLD:.4f}  ← logged as WARNING if train CE drops here\n"
    )

    # ── Datasets ──────────────────────────────────────────────────────────────
    datasets = {s: NextTokenDataset(config, split=s) for s in ["pretrain", "validation"]}

    # ── Encoder: random init — NOT warm-started from spectral/combined ────────
    encoder = SetTransformer(
        config["in_channels"],
        config["patch_size"],
        config["embed_dim"],
        config["num_heads"],
        config["num_layers"],
        pooling_head=config["pooling_head"],
        dropout=config["dropout"],
    )
    head = TokenHead(embed_dim=config["embed_dim"], n_clusters=n_clusters)

    if device.type == "cuda":
        encoder = torch.nn.DataParallel(encoder)
    encoder.to(device)
    head.to(device)

    n_enc = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    n_head = sum(p.numel() for p in head.parameters() if p.requires_grad)
    logger.info(f"Encoder: {n_enc/1e6:.2f}M params  |  Head: {n_head/1e3:.1f}K params")

    lr = config.get("lr", 1e-3)
    wd = config.get("weight_decay", 1e-4)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=lr,
        weight_decay=wd,
    )

    max_epochs = config.get("epochs", 100)
    patience = config.get("patience", 10)
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]

    epoch_resume = 0
    best_loss = math.inf
    patience_counter = 0

    ckpt_file = os.path.join(output, "checkpoint.pt")
    best_file = os.path.join(output, "best.pt")

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if os.path.isfile(ckpt_file):
        ckpt = torch.load(ckpt_file, map_location=device)
        if _has_nan_or_inf(ckpt.get("state_dict", {})):
            logger.warning("checkpoint.pt has NaN/Inf weights")
            if os.path.isfile(best_file):
                logger.warning("  → falling back to best.pt")
                ckpt = torch.load(best_file, map_location=device)
            else:
                logger.warning("  → no best.pt found, starting fresh")
                ckpt = None
        if ckpt is not None:
            encoder.load_state_dict(ckpt["state_dict"])
            if "head_state_dict" in ckpt:
                head.load_state_dict(ckpt["head_state_dict"])
            optimizer.load_state_dict(ckpt["optim_dict"])
            epoch_resume = ckpt["epoch"] + 1
            best_loss = ckpt["best_loss"]
            patience_counter = ckpt.get("patience_counter", 0)
            logger.info(f"Resumed from epoch {epoch_resume}, best_loss={best_loss:.6f}")
        else:
            logger.info("Starting fresh (checkpoint discarded)")
    else:
        logger.info("Starting fresh")

    # ── TSV log ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.join(output, "log"), exist_ok=True)
    log_tsv = os.path.join(
        output, "log", f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}.tsv"
    )
    with open(log_tsv, "w") as f:
        f.write("Epoch\tSplit\tCE_Loss\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(epoch_resume, max_epochs):
        logger.info(f"Epoch {epoch}/{max_epochs - 1}")

        train_loader = torch.utils.data.DataLoader(
            datasets["pretrain"], batch_size=batch_size,
            num_workers=num_workers, shuffle=True, drop_last=True,
        )
        train_loss = run_epoch(
            train_loader, encoder, head, optimizer, device, "pretrain", n_clusters, epoch
        )

        val_loader = torch.utils.data.DataLoader(
            datasets["validation"], batch_size=batch_size,
            num_workers=num_workers, shuffle=True,
        )
        val_loss = run_epoch(
            val_loader, encoder, head, None, device, "validation", n_clusters, epoch
        )

        with open(log_tsv, "a") as f:
            f.write(f"{epoch}\ttrain\t{train_loss:.6f}\n")
            f.write(f"{epoch}\tval\t{val_loss:.6f}\n")

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        save = {
            "epoch": epoch,
            "state_dict": encoder.state_dict(),
            "head_state_dict": head.state_dict(),
            "best_loss": best_loss,
            "optim_dict": optimizer.state_dict(),
            "patience_counter": patience_counter,
        }
        torch.save(save, ckpt_file)
        save_data(config, os.path.join(output, "config.json"))
        if is_best:
            torch.save(save, best_file)

        suffix = " [best]" if is_best else f" (patience {patience_counter}/{patience})"
        logger.info(f"Epoch {epoch}: train={train_loss:.6f} val={val_loss:.6f}{suffix}")

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    logger.info(f"Done. Best val CE: {best_loss:.6f}")


if __name__ == "__main__":
    pretrain_nexttoken()
