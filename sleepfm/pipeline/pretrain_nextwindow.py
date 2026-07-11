"""Next-window prediction pretraining for SleepFM SetTransformer encoder.

Each BAS channel pair of consecutive 5-minute windows (t, t+1) is processed
independently through the encoder. The online encoder encodes window t; a
prediction head maps that embedding to predict the target encoder's embedding
of window t+1. The target encoder is a frozen EMA copy of the online encoder
(BYOL-style) — stop-gradient prevents collapse.
"""

import copy
import datetime
import math
import os
import random
import sys

import click
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from loguru import logger
from torch import nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.models import SetTransformer
from utils import load_config, load_data, save_data


def _has_nan_or_inf(state_dict):
    return any(
        torch.isnan(v).any().item() or torch.isinf(v).any().item()
        for v in state_dict.values()
        if isinstance(v, torch.Tensor)
    )


class PredictionHead(nn.Module):
    """Maps online encoder output to predicted next-window embedding space."""

    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class NextWindowDataset(torch.utils.data.Dataset):
    """HDF5 dataset yielding consecutive 5-minute window pairs.

    Each item is a (current, next_win, mask) triple for one chunk pair
    from a single subject file. BAS channels are stacked and padded to
    max_channels.

    Returns:
        current:  [max_channels, T]  float32, window at time t
        next_win: [max_channels, T]  float32, window at time t+1
        mask:     [max_channels]     float32, 1=padded 0=real
    """

    def __init__(self, config, channel_groups, split="pretrain"):
        self.samples_per_chunk = config["sampling_duration"] * 60 * config["sampling_freq"]
        self.modality = config["nextwindow_modality"]
        self.max_channels = config[f"{self.modality}_CHANNELS"]
        self.channel_like = set(channel_groups[self.modality])

        all_paths = load_data(config["split_path"])[split]
        hdf5_paths = [os.path.join(config["data_path"], p) for p in all_paths]

        if split == "pretrain":
            random.shuffle(hdf5_paths)
        if config.get("max_files"):
            hdf5_paths = hdf5_paths[: config["max_files"]]
        if split == "validation":
            hdf5_paths = hdf5_paths[: config["val_size"]]

        self.index_map = self._build_index(hdf5_paths)
        logger.info(
            f"NextWindowDataset [{split}]: {len(self.index_map)} pairs "
            f"from {len(hdf5_paths)} files"
        )

    def _build_index(self, paths):
        index_map = []
        for path in paths:
            try:
                with h5py.File(path, "r", rdcc_nbytes=300 * 512 * 8 * 2) as hf:
                    ch_names = [
                        k for k in hf.keys()
                        if k in self.channel_like and isinstance(hf[k], h5py.Dataset)
                    ]
                    if not ch_names:
                        continue
                    n_samples = hf[ch_names[0]].shape[0]
                    n_chunks = n_samples // self.samples_per_chunk
                    # index pairs (t, t+1); need at least 2 chunks
                    for i in range(n_chunks - 1):
                        t_start = i * self.samples_per_chunk
                        t1_start = (i + 1) * self.samples_per_chunk
                        index_map.append((path, ch_names, t_start, t1_start))
            except (OSError, AttributeError):
                pass
        return index_map

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_path, ch_names, t_start, t1_start = self.index_map[idx]
        C_actual = min(len(ch_names), self.max_channels)

        current = np.zeros((self.max_channels, self.samples_per_chunk), dtype=np.float32)
        next_win = np.zeros((self.max_channels, self.samples_per_chunk), dtype=np.float32)
        mask = np.ones(self.max_channels, dtype=np.float32)
        mask[:C_actual] = 0.0

        with h5py.File(file_path, "r", rdcc_nbytes=300 * 512 * 8 * 2) as hf:
            for i, name in enumerate(ch_names[:C_actual]):
                current[i] = hf[name][t_start : t_start + self.samples_per_chunk]
                next_win[i] = hf[name][t1_start : t1_start + self.samples_per_chunk]

        # Guard NaN/Inf before they can reach the encoder or corrupt BN stats
        current = np.nan_to_num(current, nan=0.0, posinf=0.0, neginf=0.0)
        next_win = np.nan_to_num(next_win, nan=0.0, posinf=0.0, neginf=0.0)

        return (
            torch.from_numpy(current),
            torch.from_numpy(next_win),
            torch.from_numpy(mask),
        )


def collate_fn_nw(batch):
    current = torch.stack([b[0] for b in batch])   # [B, C, T]
    next_win = torch.stack([b[1] for b in batch])  # [B, C, T]
    mask = torch.stack([b[2] for b in batch])       # [B, C]
    return current, next_win, mask


@torch.no_grad()
def update_ema(online_encoder, target_encoder, momentum):
    """EMA update: target = momentum * target + (1-momentum) * online."""
    online_params = (
        online_encoder.module.parameters()
        if isinstance(online_encoder, torch.nn.DataParallel)
        else online_encoder.parameters()
    )
    for p_online, p_target in zip(online_params, target_encoder.parameters()):
        p_target.data.mul_(momentum).add_(p_online.data, alpha=1.0 - momentum)


def run_epoch(
    loader, online_encoder, target_encoder, prediction_head,
    optimizer, device, split, ema_momentum,
):
    is_train = split == "pretrain"
    online_encoder.train(is_train)
    prediction_head.train(is_train)
    target_encoder.eval()  # always eval; updated only via EMA, never backprop

    total_loss = 0.0
    total_n = 0

    with torch.set_grad_enabled(is_train):
        with tqdm.tqdm(total=len(loader), desc=f"[{split}]") as pbar:
            for current, next_win, mask in loader:
                try:
                    current = current.to(device)
                    next_win = next_win.to(device)
                    mask = mask.to(device, dtype=torch.bool)  # True = padded

                    # Guard any NaN/Inf that slipped through the dataset
                    current = torch.nan_to_num(current, nan=0.0, posinf=0.0, neginf=0.0)
                    next_win = torch.nan_to_num(next_win, nan=0.0, posinf=0.0, neginf=0.0)

                    B, C, T = current.shape
                    x_cur = current.view(B * C, 1, T)
                    x_nxt = next_win.view(B * C, 1, T)
                    # fake 1-channel mask (spatial pooling degenerates to mean)
                    ch_mask = torch.zeros(B * C, 1, device=device, dtype=torch.bool)

                    online_emb, _ = online_encoder(x_cur, ch_mask)  # [B*C, E]
                    pred = prediction_head(online_emb)               # [B*C, E]

                    with torch.no_grad():
                        target_emb, _ = target_encoder(x_nxt, ch_mask)  # [B*C, E]

                    E = online_emb.shape[-1]
                    # eps=1e-8 prevents instability when embedding norm is near zero
                    pred_n = F.normalize(pred, dim=-1, eps=1e-8).view(B, C, E)
                    target_n = F.normalize(target_emb.detach(), dim=-1, eps=1e-8).view(B, C, E)
                    # Guard NaN from any remaining zero-norm embeddings
                    pred_n = torch.nan_to_num(pred_n, nan=0.0)
                    target_n = torch.nan_to_num(target_n, nan=0.0)

                    # per-channel MSE of normalized embeddings; ignore padded channels
                    valid = (~mask).float().unsqueeze(-1)              # [B, C, 1]
                    mse_per_ch = ((pred_n - target_n) ** 2).mean(dim=-1, keepdim=True)
                    n_valid = valid.sum().clamp(min=1)
                    loss = (mse_per_ch * valid).sum() / n_valid

                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"[{split}] NaN/Inf loss — skipping batch")
                        pbar.update()
                        continue

                    if is_train:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        update_ema(online_encoder, target_encoder, ema_momentum)

                    total_loss += loss.item() * float(n_valid.item())
                    total_n += int(n_valid.item())
                except Exception as exc:
                    logger.warning(f"[{split}] batch error ({exc}) — skipping")

                pbar.set_postfix_str(f"loss={total_loss / max(total_n, 1):.5f}")
                pbar.update()

    return total_loss / max(total_n, 1)


@click.command("pretrain_nextwindow")
@click.option("--config_path", type=str, default="sleepfm/configs/config_pretrain_nextwindow.yaml")
@click.option("--channel_groups_path", type=str, default="sleepfm/configs/channel_groups.json")
@click.option("--checkpoint_path", type=str, default=None)
def pretrain_nextwindow(config_path, channel_groups_path, checkpoint_path):
    config = load_config(config_path)
    channel_groups = load_data(channel_groups_path)

    if checkpoint_path:
        output = checkpoint_path
        config = load_config(os.path.join(output, "config.json"))
    else:
        output = os.path.join(
            config["save_path"],
            f"SetTransformer/nextwindow_{config['embed_dim']}_patch_size_{config['patch_size']}",
        )
        os.makedirs(output, exist_ok=True)

    logger.info(f"Output: {output}")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Device: {device}")

    dataset = {
        split: NextWindowDataset(config, channel_groups, split=split)
        for split in ["pretrain", "validation"]
    }

    online_encoder = SetTransformer(
        config["in_channels"],
        config["patch_size"],
        config["embed_dim"],
        config["num_heads"],
        config["num_layers"],
        pooling_head=config["pooling_head"],
        dropout=config["dropout"],
    )
    prediction_head = PredictionHead(config["embed_dim"])

    if device.type == "cuda":
        online_encoder = torch.nn.DataParallel(online_encoder)
    online_encoder.to(device)
    prediction_head.to(device)

    # target encoder: frozen deep copy of the unwrapped online encoder
    raw_enc = (
        online_encoder.module
        if isinstance(online_encoder, torch.nn.DataParallel)
        else online_encoder
    )
    target_encoder = copy.deepcopy(raw_enc)
    target_encoder.to(device)
    for p in target_encoder.parameters():
        p.requires_grad_(False)

    n_online = sum(p.numel() for p in online_encoder.parameters() if p.requires_grad)
    n_head = sum(p.numel() for p in prediction_head.parameters() if p.requires_grad)
    n_target = sum(p.numel() for p in target_encoder.parameters())
    logger.info(
        f"Online encoder: {n_online / 1e6:.2f}M params  |  "
        f"Prediction head: {n_head / 1e3:.1f}K params  |  "
        f"Target encoder: {n_target / 1e6:.2f}M params (no grad)"
    )

    optimizer = torch.optim.Adam(
        list(online_encoder.parameters()) + list(prediction_head.parameters()),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.0),
    )

    ema_momentum = config.get("ema_momentum", 0.996)
    max_epochs = config.get("max_epochs", 100)
    patience = config.get("patience", 10)
    num_workers = config["num_workers"]
    batch_size = config["batch_size"]

    epoch_resume = 0
    best_loss = math.inf
    patience_counter = 0

    ckpt_file = os.path.join(output, "checkpoint.pt")
    best_file = os.path.join(output, "best.pt")
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
            online_encoder.load_state_dict(ckpt["state_dict"])
            prediction_head.load_state_dict(ckpt["head_state_dict"])
            target_encoder.load_state_dict(ckpt["target_state_dict"])
            optimizer.load_state_dict(ckpt["optim_dict"])
            epoch_resume = ckpt["epoch"] + 1
            best_loss = ckpt["best_loss"]
            patience_counter = ckpt.get("patience_counter", 0)
            logger.info(f"Resumed from epoch {epoch_resume}, best_loss={best_loss:.6f}")
        else:
            logger.info("Starting fresh (checkpoint discarded)")
    else:
        logger.info("Starting fresh")

    os.makedirs(os.path.join(output, "log"), exist_ok=True)
    log_tsv = os.path.join(
        output, "log", f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}.tsv"
    )
    with open(log_tsv, "w") as f:
        f.write("Epoch\tSplit\tLoss\n")

    for epoch in range(epoch_resume, max_epochs):
        logger.info(f"Epoch {epoch}/{max_epochs - 1}")

        train_loader = torch.utils.data.DataLoader(
            dataset["pretrain"],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn_nw,
        )
        train_loss = run_epoch(
            train_loader, online_encoder, target_encoder, prediction_head,
            optimizer, device, "pretrain", ema_momentum,
        )
        logger.info(f"  train loss: {train_loss:.6f}")

        val_loader = torch.utils.data.DataLoader(
            dataset["validation"],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=collate_fn_nw,
        )
        val_loss = run_epoch(
            val_loader, online_encoder, target_encoder, prediction_head,
            None, device, "validation", ema_momentum,
        )
        logger.info(f"  val   loss: {val_loss:.6f}")

        with open(log_tsv, "a") as f:
            f.write(f"{epoch}\ttrain\t{train_loss:.6f}\n")
            f.write(f"{epoch}\tval\t{val_loss:.6f}\n")

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"  no improvement ({patience_counter}/{patience})")

        save = {
            "epoch": epoch,
            "state_dict": online_encoder.state_dict(),
            "head_state_dict": prediction_head.state_dict(),
            "target_state_dict": target_encoder.state_dict(),
            "optim_dict": optimizer.state_dict(),
            "best_loss": best_loss,
            "loss": val_loss,
            "patience_counter": patience_counter,
        }
        torch.save(save, ckpt_file)
        save_data(config, os.path.join(output, "config.json"))

        if is_best:
            torch.save(save, os.path.join(output, "best.pt"))

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    logger.info(f"Done. Best val loss: {best_loss:.6f}")


if __name__ == "__main__":
    pretrain_nextwindow()
