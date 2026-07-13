"""Spectral reconstruction pretraining for SleepFM SetTransformer encoder.

BAS channels (EEG1, EEG2, EEG3, EOG-L, EOG-R) are processed per-channel
through the encoder. A small MLP decoder predicts log10 mean band power in
5 EEG spectral bands (delta/theta/alpha/sigma/beta) per channel.

Targets use log10(power + 1e-8) rather than log1p, giving values in the
range roughly -5 to -1 that the decoder can meaningfully learn to predict.
"""

import datetime
import math
import os
import sys

import click
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from loguru import logger
from scipy.signal import welch
from torch import nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.models import SetTransformer
from utils import load_config, load_data, save_data


BAS_CHANNELS = ["EEG1", "EEG2", "EEG3", "EOG-L", "EOG-R"]
WINDOW_SIZE = 640   # 5 seconds at 128 Hz
FS = 128
DEFAULT_BANDS = [(0.5, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 15.0), (15.0, 30.0)]


def _has_nan_or_inf(state_dict):
    return any(
        torch.isnan(v).any().item() or torch.isinf(v).any().item()
        for v in state_dict.values()
        if isinstance(v, torch.Tensor)
    )


class SpectralDecoder(nn.Module):
    """MLP: Linear(embed_dim, 128) -> ReLU -> Linear(128, 5)."""

    def __init__(self, embed_dim=128, n_bands=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_bands),
        )

    def forward(self, x):
        return self.net(x)


class SpectralDataset(torch.utils.data.Dataset):
    """HDF5 dataset yielding (signal_window, spectral_targets, channel_mask).

    signal_window:     [max_channels, 640]  float32  (zero-padded)
    spectral_targets:  [max_channels, 5]   float32  log10(power + 1e-8)
    channel_mask:      [max_channels]       bool     True = padded channel
    """

    def __init__(self, config, split="pretrain"):
        self.window_size = WINDOW_SIZE
        self.max_channels = config.get("max_channels", 10)
        raw_bands = config.get("spectral_bands", DEFAULT_BANDS)
        self.bands = [tuple(b) for b in raw_bands]

        all_paths = load_data(config["split_path"])[split]
        hdf5_paths = [os.path.join(config["data_path"], p) for p in all_paths]
        if split == "validation":
            hdf5_paths = hdf5_paths[:config.get("val_size", 100)]

        self.index_map = self._build_index(hdf5_paths)
        logger.info(
            f"SpectralDataset [{split}]: {len(self.index_map)} windows "
            f"from {len(hdf5_paths)} files"
        )

    def _build_index(self, paths):
        index_map = []
        for path in paths:
            try:
                with h5py.File(path, "r") as hf:
                    available = [ch for ch in BAS_CHANNELS if ch in hf]
                    if not available:
                        continue
                    n_samples = hf[available[0]].shape[0]
                    n_windows = n_samples // self.window_size
                    for i in range(n_windows):
                        index_map.append((path, available, i * self.window_size))
            except (OSError, AttributeError):
                pass
        return index_map

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_path, available, window_start = self.index_map[idx]

        raw = np.zeros((self.max_channels, self.window_size), dtype=np.float32)
        spectral = np.zeros((self.max_channels, 5), dtype=np.float32)
        mask = np.ones(self.max_channels, dtype=bool)  # True = padded

        with h5py.File(file_path, "r") as hf:
            for i, ch in enumerate(available[:self.max_channels]):
                sig = hf[ch][window_start:window_start + self.window_size].astype(np.float32)
                sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
                raw[i] = sig
                spectral[i] = self._compute_spectral_log10(sig)
                mask[i] = False

        return (
            torch.from_numpy(raw),
            torch.from_numpy(spectral),
            torch.from_numpy(mask),
        )

    def _compute_spectral_log10(self, signal):
        """log10(mean_band_power + 1e-8) for each of 5 spectral bands."""
        freqs, psd = welch(signal, fs=FS, nperseg=128)
        targets = np.zeros(5, dtype=np.float32)
        for j, (lo, hi) in enumerate(self.bands):
            m = (freqs >= lo) & (freqs < hi)
            power = float(psd[m].mean()) if m.any() else 0.0
            if np.isnan(power) or np.isinf(power):
                power = 0.0
            targets[j] = np.log10(power + 1e-8)
        return targets


def run_epoch(loader, encoder, decoder, optimizer, device, split):
    is_train = split == "pretrain"
    if not is_train:
        encoder.eval()
        decoder.eval()
    else:
        encoder.train()
        decoder.train()

    total_loss = 0.0
    total_n = 0

    with torch.set_grad_enabled(is_train):
        with tqdm.tqdm(total=len(loader), desc=f"[{split}]") as pbar:
            for raw_data, spectral_targets, mask in loader:
                try:
                    raw_data = raw_data.to(device)
                    spectral_targets = spectral_targets.to(device)
                    mask = mask.to(device)  # bool, True = padded

                    B, C, T = raw_data.shape
                    x = raw_data.view(B * C, 1, T)
                    ch_mask = torch.zeros(B * C, 1, device=device, dtype=torch.bool)

                    emb, _ = encoder(x, ch_mask)        # [B*C, embed_dim]
                    pred = decoder(emb).view(B, C, -1)  # [B, C, 5]

                    # masked MSE — ignore padded channels
                    valid = (~mask).float().unsqueeze(-1)  # [B, C, 1]
                    sq_err = ((pred - spectral_targets) ** 2) * valid
                    n_valid = valid.sum().clamp(min=1)
                    loss = sq_err.sum() / n_valid

                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"[{split}] NaN/Inf loss — skipping batch")
                        pbar.update()
                        continue

                    if is_train:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    total_loss += loss.item() * float(n_valid.item())
                    total_n += int(n_valid.item())
                except Exception as exc:
                    logger.warning(f"[{split}] batch error ({exc}) — skipping")

                pbar.set_postfix_str(f"loss={total_loss / max(total_n, 1):.5f}")
                pbar.update()

    if not is_train:
        encoder.train()
        decoder.train()

    return total_loss / max(total_n, 1)


@click.command("pretrain_spectral")
@click.option("--config_path", default="sleepfm/configs/config_pretrain_spectral.yaml")
@click.option("--channel_groups_path", default="sleepfm/configs/channel_groups.json")
@click.option("--checkpoint_path", default=None)
def pretrain_spectral(config_path, channel_groups_path, checkpoint_path):
    config = load_config(config_path)

    if checkpoint_path:
        output = checkpoint_path
        config = load_config(os.path.join(output, "config.json"))
    else:
        output = os.path.join(
            config["save_path"],
            f"SetTransformer/spectral_{config['embed_dim']}_patch_size_{config['patch_size']}",
        )
        os.makedirs(output, exist_ok=True)

    logger.info(f"Output: {output}")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Device: {device}")

    dataset = {s: SpectralDataset(config, split=s) for s in ["pretrain", "validation"]}

    encoder = SetTransformer(
        config["in_channels"],
        config["patch_size"],
        config["embed_dim"],
        config["num_heads"],
        config["num_layers"],
        pooling_head=config["pooling_head"],
        dropout=config["dropout"],
    )
    decoder = SpectralDecoder(embed_dim=config["embed_dim"], n_bands=5)

    if device.type == "cuda":
        encoder = torch.nn.DataParallel(encoder)
    encoder.to(device)
    decoder.to(device)

    n_enc = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    n_dec = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    logger.info(f"Encoder: {n_enc/1e6:.2f}M params  |  Decoder: {n_dec/1e3:.1f}K params")

    lr = config.get("spectral_lr", config.get("lr", 1e-3))
    wd = config.get("spectral_weight_decay", config.get("weight_decay", 1e-4))
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr,
        weight_decay=wd,
    )

    max_epochs = config.get("max_epochs", 100)
    patience = config.get("patience", 10)
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]

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
            encoder.load_state_dict(ckpt["state_dict"])
            if "decoder_state_dict" in ckpt:
                decoder.load_state_dict(ckpt["decoder_state_dict"])
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
            dataset["pretrain"], batch_size=batch_size,
            num_workers=num_workers, shuffle=True, drop_last=True,
        )
        train_loss = run_epoch(train_loader, encoder, decoder, optimizer, device, "pretrain")

        val_loader = torch.utils.data.DataLoader(
            dataset["validation"], batch_size=batch_size,
            num_workers=num_workers, shuffle=False,
        )
        val_loss = run_epoch(val_loader, encoder, decoder, None, device, "validation")

        with open(log_tsv, "a") as f:
            f.write(f"{epoch}\ttrain\t{train_loss:.6f}\n")
            f.write(f"{epoch}\tval\t{val_loss:.6f}\n")

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # required keys match pretrain.py checkpoint format for generate_embeddings.py
        save = {
            "epoch": epoch,
            "state_dict": encoder.state_dict(),
            "best_loss": best_loss,
            "optim_dict": optimizer.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
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

    logger.info(f"Done. Best val loss: {best_loss:.6f}")


if __name__ == "__main__":
    pretrain_spectral()
