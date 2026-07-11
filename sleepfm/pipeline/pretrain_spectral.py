"""Spectral reconstruction pretraining for SleepFM SetTransformer encoder.

Each channel is processed independently through the encoder; a small MLP
decoder predicts the log1p mean band power in 5 EEG spectral bands.
"""

import datetime
import math
import os
import random
import sys
import time

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

SPECTRAL_BANDS = [
    (0.5, 4.0),   # Delta
    (4.0, 8.0),   # Theta
    (8.0, 12.0),  # Alpha
    (12.0, 15.0), # Sigma
    (15.0, 30.0), # Beta
]


class SpectralDecoder(nn.Module):
    def __init__(self, embed_dim, n_bands=5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_bands),
        )

    def forward(self, x):
        return self.mlp(x)


class SpectralDataset(torch.utils.data.Dataset):
    """HDF5 dataset that yields (raw_signal, spectral_targets, channel_mask).

    raw_signal:       [max_channels, T]  float32, padded with zeros
    spectral_targets: [max_channels, 5]  float32, log1p mean band power
    channel_mask:     [max_channels]     float32, 1=padded 0=real
    """

    def __init__(self, config, channel_groups, hdf5_paths=None, split="pretrain"):
        self.fs = config["sampling_freq"]
        self.patch_size = config["patch_size"]
        self.samples_per_chunk = config["sampling_duration"] * 60 * config["sampling_freq"]
        self.modality = config["spectral_modality"]
        self.max_channels = config[f"{self.modality}_CHANNELS"]
        self.channel_like = set(channel_groups[self.modality])
        self.bands = [tuple(b) for b in config.get("spectral_bands", SPECTRAL_BANDS)]

        if hdf5_paths is None:
            all_paths = load_data(config["split_path"])[split]
            hdf5_paths = [os.path.join(config["data_path"], p) for p in all_paths]

        if split == "pretrain":
            random.shuffle(hdf5_paths)
        if config.get("max_files"):
            hdf5_paths = hdf5_paths[: config["max_files"]]
        if split == "validation":
            hdf5_paths = hdf5_paths[: config["val_size"]]

        self.index_map = self._build_index(hdf5_paths)
        logger.info(f"SpectralDataset [{split}]: {len(self.index_map)} chunks from {len(hdf5_paths)} files")

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
                    for i in range(n_chunks):
                        index_map.append((path, ch_names, i * self.samples_per_chunk))
            except (OSError, AttributeError):
                pass
        return index_map

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_path, ch_names, chunk_start = self.index_map[idx]
        C_actual = min(len(ch_names), self.max_channels)

        raw = np.zeros((self.max_channels, self.samples_per_chunk), dtype=np.float32)
        mask = np.ones(self.max_channels, dtype=np.float32)
        mask[:C_actual] = 0.0

        with h5py.File(file_path, "r", rdcc_nbytes=300 * 512 * 8 * 2) as hf:
            for i, name in enumerate(ch_names[:C_actual]):
                raw[i] = hf[name][chunk_start : chunk_start + self.samples_per_chunk]

        spectral = self._compute_spectral(raw[:C_actual])
        spectral_padded = np.zeros((self.max_channels, 5), dtype=np.float32)
        spectral_padded[:C_actual] = spectral

        return (
            torch.from_numpy(raw),
            torch.from_numpy(spectral_padded),
            torch.from_numpy(mask),
        )

    def _compute_spectral(self, signal):
        C, T = signal.shape
        n_patches = T // self.patch_size
        targets = np.zeros((C, 5), dtype=np.float32)
        for c in range(C):
            window_powers = []
            for w in range(n_patches):
                patch = signal[c, w * self.patch_size : (w + 1) * self.patch_size]
                freqs, psd = welch(patch, fs=self.fs, nperseg=128)
                bp = []
                for lo, hi in self.bands:
                    m = (freqs >= lo) & (freqs < hi)
                    bp.append(float(psd[m].mean()) if m.any() else 0.0)
                window_powers.append(bp)
            targets[c] = np.log1p(np.mean(window_powers, axis=0))
        return targets


def run_epoch(loader, encoder, decoder, optimizer, device, split):
    is_train = split == "pretrain"
    encoder.train(is_train)
    decoder.train(is_train)

    total_loss = 0.0
    total_n = 0

    with torch.set_grad_enabled(is_train):
        with tqdm.tqdm(total=len(loader), desc=f"[{split}]") as pbar:
            for raw_data, spectral_targets, mask in loader:
                raw_data = raw_data.to(device)
                spectral_targets = spectral_targets.to(device)
                mask = mask.to(device, dtype=torch.bool)

                B, C, T = raw_data.shape

                # process each channel independently as a 1-channel modality
                x = raw_data.view(B * C, 1, T)
                ch_mask = torch.zeros(B * C, 1, device=device, dtype=torch.bool)

                emb, _ = encoder(x, ch_mask)  # [B*C, embed_dim]
                pred = decoder(emb).view(B, C, -1)  # [B, C, 5]

                # ignore padded channels
                valid = (~mask).float().unsqueeze(-1)  # [B, C, 1]
                sq_err = ((pred - spectral_targets) ** 2) * valid
                n_valid = valid.sum().clamp(min=1)
                loss = sq_err.sum() / n_valid

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * float(n_valid.item())
                total_n += int(n_valid.item())
                pbar.set_postfix_str(f"loss={total_loss / max(total_n, 1):.5f}")
                pbar.update()

    return total_loss / max(total_n, 1)


@click.command("pretrain_spectral")
@click.option("--config_path", type=str, default="sleepfm/configs/config_pretrain_spectral.yaml")
@click.option("--channel_groups_path", type=str, default="sleepfm/configs/channel_groups.json")
@click.option("--checkpoint_path", type=str, default=None)
def pretrain_spectral(config_path, channel_groups_path, checkpoint_path):
    config = load_config(config_path)
    channel_groups = load_data(channel_groups_path)

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

    dataset = {
        split: SpectralDataset(config, channel_groups, split=split)
        for split in ["pretrain", "validation"]
    }

    encoder = SetTransformer(
        config["in_channels"],
        config["patch_size"],
        config["embed_dim"],
        config["num_heads"],
        config["num_layers"],
        pooling_head=config["pooling_head"],
        dropout=config["dropout"],
    )
    decoder = SpectralDecoder(config["embed_dim"], n_bands=5)

    if device.type == "cuda":
        encoder = torch.nn.DataParallel(encoder)
    encoder.to(device)
    decoder.to(device)

    total_enc = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total_dec = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    logger.info(f"Encoder params: {total_enc / 1e6:.2f}M  |  Decoder params: {total_dec / 1e6:.2f}M")

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.0),
    )

    max_epochs = config.get("max_epochs", 100)
    patience = config.get("patience", 10)
    num_workers = config["num_workers"]
    batch_size = config["batch_size"]

    epoch_resume = 0
    best_loss = math.inf
    patience_counter = 0

    ckpt_file = os.path.join(output, "checkpoint.pt")
    if os.path.isfile(ckpt_file):
        ckpt = torch.load(ckpt_file, map_location=device)
        encoder.load_state_dict(ckpt["state_dict"])
        decoder.load_state_dict(ckpt["decoder_state_dict"])
        optimizer.load_state_dict(ckpt["optim_dict"])
        epoch_resume = ckpt["epoch"] + 1
        best_loss = ckpt["best_loss"]
        patience_counter = ckpt.get("patience_counter", 0)
        logger.info(f"Resumed from epoch {epoch_resume}, best_loss={best_loss:.6f}")
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
        )
        train_loss = run_epoch(train_loader, encoder, decoder, optimizer, device, "pretrain")
        logger.info(f"  train loss: {train_loss:.6f}")

        val_loader = torch.utils.data.DataLoader(
            dataset["validation"],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        val_loss = run_epoch(val_loader, encoder, decoder, None, device, "validation")
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
            "state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
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
    pretrain_spectral()
