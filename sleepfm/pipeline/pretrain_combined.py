"""Combined (contrastive + spectral) pretraining for SleepFM SetTransformer encoder.

Adds a spectral reconstruction auxiliary loss on top of the standard
leave_one_out contrastive objective from pretrain.py.

total_loss = contrastive_loss + lambda_spectral * spectral_loss

The spectral loss predicts log10(mean_band_power + 1e-8) for 5 EEG frequency
bands per BAS channel, using a randomly sampled 640-sample (5-second) window
from within each 5-minute contrastive chunk. Targets are read from the
pre-built disk cache (see scripts/build_spectral_cache.py).
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
from torch import nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.models import SetTransformer
from models.dataset import SetTransformerDataset, collate_fn
from utils import load_config, load_data, save_data, count_parameters


BAS_CHANNELS = ["EEG1", "EEG2", "EEG3", "EOG-L", "EOG-R"]
SPECTRAL_WINDOW = 640   # 5 seconds at 128 Hz
MAX_CHANNELS = 10


def _has_nan_or_inf(state_dict):
    return any(
        torch.isnan(v).any().item() or torch.isinf(v).any().item()
        for v in state_dict.values()
        if isinstance(v, torch.Tensor)
    )


# ── SpectralDecoder ──────────────────────────────────────────────────────────

class SpectralDecoder(nn.Module):
    """MLP: Linear(embed_dim, 128) -> ReLU -> Linear(128, 5).  Same as pretrain_spectral.py."""

    def __init__(self, embed_dim=128, n_bands=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_bands),
        )

    def forward(self, x):
        return self.net(x)


# ── SpectralCache ────────────────────────────────────────────────────────────

class SpectralCache:
    """Read-only view of the spectral cache built by build_spectral_cache.py.

    Lookup: (file_path, chunk_start_in_samples) -> (signals [MAX_CH,640],
            targets [MAX_CH,5], mask [MAX_CH]) where mask[c]=True means padded.

    chunk_start must be a multiple of SPECTRAL_WINDOW (640).  All contrastive
    chunk_starts are multiples of 38400 = 60 * 640, so they map directly to
    cache entries at the first 5-second window of each 5-minute chunk.
    """

    def __init__(self, config, split="pretrain"):
        cache_dir = config.get("spectral_cache_dir", "/scratch/project_2019517/sleepfm-data")
        signals_path = os.path.join(cache_dir, f"spectral_signals_{split}.bin")
        cache_path = os.path.join(cache_dir, f"spectral_cache_{split}.npz")

        if not os.path.isfile(signals_path) or not os.path.isfile(cache_path):
            raise FileNotFoundError(
                f"Spectral cache not found for split='{split}'.\n"
                f"  Expected: {signals_path}\n"
                f"  Run first: sbatch scripts/run_build_spectral_cache.slurm"
            )

        cache = np.load(cache_path)
        self.targets = cache["targets"]   # [N, MAX_CH, 5] float32
        self.masks = cache["masks"]       # [N, MAX_CH]   bool
        N = int(cache["n_windows"])

        self.signals = np.memmap(
            signals_path, dtype="float32", mode="r",
            shape=(N, MAX_CHANNELS, SPECTRAL_WINDOW),
        )

        # Fast path: build_spectral_cache.py (updated) saves file_paths + window_starts
        if "file_paths" in cache and "window_starts" in cache:
            fps = [p.decode() for p in cache["file_paths"]]
            wss = cache["window_starts"]
            self._lookup = {(fp, int(ws)): i for i, (fp, ws) in enumerate(zip(fps, wss))}
            logger.info(f"SpectralCache [{split}]: {N} windows via saved index")
        else:
            # Fallback: scan HDF5 files for shape metadata only (fast — no signal reads)
            logger.info(f"SpectralCache [{split}]: saved index missing, scanning HDF5 metadata ...")
            all_names = load_data(config["split_path"])[split]
            hdf5_paths = [os.path.join(config["data_path"], p) for p in all_names]
            if split == "validation":
                hdf5_paths = hdf5_paths[:config.get("val_size", 100)]
            self._lookup = {}
            cache_idx = 0
            for path in hdf5_paths:
                try:
                    with h5py.File(path, "r") as hf:
                        available = [ch for ch in BAS_CHANNELS if ch in hf]
                        if not available:
                            continue
                        n_samples = hf[available[0]].shape[0]
                        n_windows = n_samples // SPECTRAL_WINDOW
                        for i in range(n_windows):
                            self._lookup[(path, i * SPECTRAL_WINDOW)] = cache_idx
                            cache_idx += 1
                except Exception as exc:
                    logger.warning(f"SpectralCache scan: skipping {path}: {exc}")
            logger.info(f"SpectralCache [{split}]: scanned {len(self._lookup)} windows")

    def get(self, file_path, chunk_start):
        """Return numpy arrays (signals, targets, mask) or (None, None, None) on miss."""
        idx = self._lookup.get((file_path, chunk_start))
        if idx is None:
            return None, None, None
        return (
            np.array(self.signals[idx], dtype=np.float32),
            self.targets[idx].copy(),
            self.masks[idx].copy(),
        )


# ── CombinedDataset & collate ────────────────────────────────────────────────

class CombinedDataset(torch.utils.data.Dataset):
    """SetTransformerDataset with spectral targets attached from SpectralCache."""

    def __init__(self, config, channel_groups, split="pretrain", spectral_cache=None):
        self._base = SetTransformerDataset(config, channel_groups, split=split)
        self._cache = spectral_cache

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        target_list, file_path, dset_names, chunk_start, mod_len = self._base[idx]

        spec_sig, spec_tgt, spec_msk = (None, None, None)
        if self._cache is not None:
            # Pick a random 5-second window within the 5-minute chunk so that
            # the spectral loss sees all 60 windows uniformly rather than always
            # using only the first one (which wastes 59/60 of the cache).
            offset = random.randrange(60) * SPECTRAL_WINDOW
            spec_sig, spec_tgt, spec_msk = self._cache.get(file_path, chunk_start + offset)

        return target_list, file_path, dset_names, chunk_start, mod_len, spec_sig, spec_tgt, spec_msk


def combined_collate_fn(batch):
    """Calls the standard contrastive collate_fn and stacks spectral tensors."""
    spec_sigs = [item[5] for item in batch]
    spec_tgts = [item[6] for item in batch]
    spec_msks = [item[7] for item in batch]

    # Reformat to the 5-tuple that collate_fn expects
    base_batch = [(item[0], item[1], item[2], item[3], item[4]) for item in batch]
    contrastive_out = collate_fn(base_batch)
    # contrastive_out = (padded_batch_list, mask_list, file_paths, dset_names_list, chunk_starts)

    has_spec = all(s is not None for s in spec_sigs)
    if has_spec:
        spec_sigs_t = torch.from_numpy(np.stack(spec_sigs))  # [B, MAX_CH, 640]
        spec_tgts_t = torch.from_numpy(np.stack(spec_tgts))  # [B, MAX_CH, 5]
        spec_msks_t = torch.from_numpy(np.stack(spec_msks))  # [B, MAX_CH]
    else:
        spec_sigs_t = spec_tgts_t = spec_msks_t = None

    return (*contrastive_out, spec_sigs_t, spec_tgts_t, spec_msks_t)


# ── Combined iteration ───────────────────────────────────────────────────────

def run_combined_iter(
    batch, num_modalities, model, decoder, device,
    temperature, batch_size, ij, lambda_spectral,
):
    """One forward pass: leave_one_out contrastive + spectral MSE.

    Returns (total_loss, c_loss, s_loss, pairwise_loss, correct, pairs).
    total_loss is NOT yet divided by batch_size — caller does that before backward.
    """
    batch_data, mask_list, file_paths, dset_names_list, chunk_starts, \
        spec_sigs, spec_tgts, spec_msks = batch

    (bas, resp, ekg, emg) = batch_data
    (mask_bas, mask_resp, mask_ekg, mask_emg) = mask_list

    bas = bas.to(device, dtype=torch.float)
    resp = resp.to(device, dtype=torch.float)
    ekg = ekg.to(device, dtype=torch.float)
    emg = emg.to(device, dtype=torch.float)
    mask_bas = mask_bas.to(device, dtype=torch.bool)
    mask_resp = mask_resp.to(device, dtype=torch.bool)
    mask_ekg = mask_ekg.to(device, dtype=torch.bool)
    mask_emg = mask_emg.to(device, dtype=torch.bool)

    # ── contrastive embeddings ───────────────────────────────────────────────
    emb = [
        model(bas, mask_bas)[0],
        model(resp, mask_resp)[0],
        model(ekg, mask_ekg)[0],
        model(emg, mask_emg)[0],
    ]
    for i in range(num_modalities):
        emb[i] = F.normalize(emb[i])

    # ── leave_one_out loss ───────────────────────────────────────────────────
    c_loss = torch.tensor(0.0, device=device)
    pairwise_loss = np.zeros((num_modalities, 2), dtype=float)
    correct = np.zeros((num_modalities, 2), dtype=int)
    pairs = np.zeros((num_modalities, 2), dtype=int)

    for i in range(num_modalities):
        other_emb = (
            torch.stack([emb[j] for j in list(range(i)) + list(range(i + 1, num_modalities))])
            .sum(0) / (num_modalities - 1)
        )
        logits = torch.matmul(emb[i], other_emb.transpose(0, 1)) * torch.exp(temperature)
        labels = torch.arange(logits.shape[0], device=device)

        l = F.cross_entropy(logits, labels, reduction="sum")
        c_loss = c_loss + l
        pairwise_loss[i, 0] = l.item()
        correct[i, 0] = (torch.argmax(logits, dim=0) == labels).sum().item() if len(logits) else 0
        pairs[i, 0] = batch_size

        l = F.cross_entropy(logits.transpose(0, 1), labels, reduction="sum")
        c_loss = c_loss + l
        pairwise_loss[i, 1] = l.item()
        correct[i, 1] = (torch.argmax(logits, dim=1) == labels).sum().item() if len(logits) else 0
        pairs[i, 1] = batch_size

    c_loss = c_loss / (num_modalities * 2)

    # ── spectral loss ────────────────────────────────────────────────────────
    s_loss = torch.tensor(0.0, device=device)
    if spec_sigs is not None and not (torch.isnan(spec_sigs).any() or torch.isinf(spec_sigs).any()):
        spec_sigs = spec_sigs.to(device)    # [B, MAX_CH, 640]
        spec_tgts = spec_tgts.to(device)    # [B, MAX_CH, 5]
        spec_msks = spec_msks.to(device)    # [B, MAX_CH] bool

        B, C, T = spec_sigs.shape
        x = spec_sigs.view(B * C, 1, T)
        ch_mask = torch.zeros(B * C, 1, device=device, dtype=torch.bool)

        spec_emb, _ = model(x, ch_mask)    # [B*C, embed_dim]
        pred = decoder(spec_emb).view(B, C, -1)  # [B, C, 5]

        valid = (~spec_msks).float().unsqueeze(-1)   # [B, C, 1]
        sq_err = ((pred - spec_tgts) ** 2) * valid
        n_valid = valid.sum().clamp(min=1)
        s_loss = sq_err.sum() / n_valid

    total_loss = c_loss + lambda_spectral * s_loss
    return total_loss, c_loss, s_loss, pairwise_loss, correct, pairs


# ── main ─────────────────────────────────────────────────────────────────────

@click.command("pretrain_combined")
@click.option("--config_path", default="sleepfm/configs/config_pretrain_combined.yaml")
@click.option("--channel_groups_path", default="sleepfm/configs/channel_groups.json")
@click.option("--checkpoint_path", default=None)
def pretrain_combined(config_path, channel_groups_path, checkpoint_path):
    config = load_config(config_path)
    channel_groups = load_data(channel_groups_path)

    if checkpoint_path:
        output = checkpoint_path
        config = load_config(os.path.join(output, "config.json"))
    else:
        output = os.path.join(
            config["save_path"],
            f"SetTransformer/combined_{config['embed_dim']}_patch_size_{config['patch_size']}",
        )
        os.makedirs(output, exist_ok=True)

    logger.info(f"Output: {output}")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Device: {device}")

    # ── load spectral caches ─────────────────────────────────────────────────
    logger.info("Loading spectral caches ...")
    spectral_caches = {
        s: SpectralCache(config, split=s) for s in ["pretrain", "validation"]
    }

    # ── datasets ─────────────────────────────────────────────────────────────
    logger.info("Indexing contrastive datasets ...")
    t0 = time.time()
    datasets = {
        s: CombinedDataset(config, channel_groups, split=s, spectral_cache=spectral_caches[s])
        for s in ["pretrain", "validation"]
    }
    logger.info(f"Datasets indexed in {time.time() - t0:.1f}s")

    # ── model ────────────────────────────────────────────────────────────────
    model = SetTransformer(
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
        model = torch.nn.DataParallel(model)
    model.to(device)
    decoder.to(device)

    total_layers, total_params = count_parameters(model)
    n_dec = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    logger.info(f"Encoder: {total_params/1e6:.2f}M params ({total_layers} layers)")
    logger.info(f"Decoder: {n_dec/1e3:.1f}K params")

    # ── optimizer & scheduler (match pretrain.py exactly) ───────────────────
    temperature = torch.nn.parameter.Parameter(torch.as_tensor(float(config["temperature"])))
    optim_params = list(model.parameters()) + list(decoder.parameters()) + [temperature]
    optim = torch.optim.SGD(
        optim_params,
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optim,
        step_size=config["lr_step_period"] if config["lr_step_period"] else math.inf,
        gamma=config["gamma"],
    )

    lambda_spectral = float(config.get("lambda_spectral", 0.5))
    num_modalities = len(config["modality_types"])
    ij = sum([((i, j), (j, i)) for i in range(num_modalities) for j in range(i + 1, num_modalities)], ())

    epochs = config["epochs"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    patience = config.get("patience", 10)

    epoch_resume = 0
    best_loss = math.inf
    patience_counter = 0

    ckpt_file = os.path.join(output, "checkpoint.pt")
    best_file = os.path.join(output, "best.pt")

    # ── resume ───────────────────────────────────────────────────────────────
    if os.path.isfile(ckpt_file):
        ckpt = torch.load(ckpt_file, map_location=device)
        if _has_nan_or_inf(ckpt.get("state_dict", {})):
            logger.warning("checkpoint.pt has NaN/Inf; trying best.pt")
            ckpt = torch.load(best_file, map_location=device) if os.path.isfile(best_file) else None
        if ckpt is not None:
            model.load_state_dict(ckpt["state_dict"])
            if "decoder_state_dict" in ckpt:
                decoder.load_state_dict(ckpt["decoder_state_dict"])
            with torch.no_grad():
                temperature.fill_(ckpt["temperature"])
            optim.load_state_dict(ckpt["optim_dict"])
            scheduler.load_state_dict(ckpt["scheduler_dict"])
            epoch_resume = ckpt["epoch"] + 1
            best_loss = ckpt["best_loss"]
            patience_counter = ckpt.get("patience_counter", 0)
            logger.info(f"Resumed from epoch {epoch_resume}, best_loss={best_loss:.6f}")
        else:
            logger.info("Starting fresh (checkpoint discarded)")
    else:
        logger.info("Starting fresh")

    # ── log file ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.join(output, "log"), exist_ok=True)
    log_tsv = os.path.join(
        output, "log", f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}.tsv"
    )
    with open(log_tsv, "w") as f:
        f.write("Epoch\tSplit\tTotal Loss\tC Loss\tS Loss\t")
        f.write("\t".join(
            f"{config['modality_types'][i]}-other\tother-{config['modality_types'][i]}"
            for i in range(num_modalities)
        ))
        f.write("\tTemperature\n")

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(epoch_resume, epochs):
        logger.info(f"Epoch {epoch}/{epochs - 1}")

        for split in ["pretrain", "validation"]:
            is_train = split == "pretrain"
            model.train()   # always train mode — no eval() switching
            decoder.train()

            loader = torch.utils.data.DataLoader(
                datasets[split],
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=is_train,
                collate_fn=combined_collate_fn,
                drop_last=is_train,
            )

            total_loss = 0.0
            total_c_loss = 0.0
            total_s_loss = 0.0
            total_pairwise = np.zeros((num_modalities, 2), dtype=float)
            total_correct = np.zeros((num_modalities, 2), dtype=int)
            total_pairs = np.zeros((num_modalities, 2), dtype=int)
            total_n = 0

            with torch.set_grad_enabled(is_train):
                with tqdm.tqdm(total=len(loader), desc=f"[{split}]") as pbar:
                    for batch in loader:
                        try:
                            loss, c_loss, s_loss, pw_loss, corr, pairs = run_combined_iter(
                                batch, num_modalities, model, decoder, device,
                                temperature, batch_size, ij, lambda_spectral,
                            )
                        except Exception as exc:
                            logger.warning(f"[{split}] batch error: {exc} — skipping")
                            pbar.update()
                            continue

                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.warning(f"[{split}] NaN/Inf total_loss — skipping batch")
                            pbar.update()
                            continue

                        bs = batch[0][0].size(0)
                        total_loss += loss.item()
                        total_c_loss += c_loss.item()
                        total_s_loss += s_loss.item()
                        total_pairwise += pw_loss
                        total_correct += corr
                        total_pairs += pairs
                        total_n += bs

                        if is_train:
                            optim.zero_grad()
                            (loss / bs).backward()
                            optim.step()
                            if temperature < 0:
                                with torch.no_grad():
                                    temperature.fill_(0.0)

                        pbar.set_postfix_str(
                            f"loss={total_loss/max(total_n,1):.4f} "
                            f"c={total_c_loss/max(total_n,1):.4f} "
                            f"s={total_s_loss/max(total_n,1):.4f} "
                            f"T={temperature.item():.3f}"
                        )
                        pbar.update()

            epoch_loss = total_loss / max(total_n, 1)
            epoch_c = total_c_loss / max(total_n, 1)
            epoch_s = total_s_loss / max(total_n, 1)

            logger.info(
                f"Epoch {epoch} [{split}]: total={epoch_loss:.6f} "
                f"c={epoch_c:.6f} s={epoch_s:.6f}  T={temperature.item():.4f}"
            )

            with open(log_tsv, "a") as f:
                pw_vals = "\t".join(
                    f"{total_pairwise[i,0]/max(total_pairs[i,0],1):.5f}\t"
                    f"{total_pairwise[i,1]/max(total_pairs[i,1],1):.5f}"
                    for i in range(num_modalities)
                )
                f.write(f"{epoch}\t{split}\t{epoch_loss:.6f}\t{epoch_c:.6f}\t{epoch_s:.6f}\t"
                        f"{pw_vals}\t{temperature.item():.5f}\n")

            # early stopping and checkpoint on validation
            if split == "validation":
                is_best = epoch_loss < best_loss
                if is_best:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                save = {
                    "epoch": epoch,
                    "temperature": temperature.item(),
                    "optim_dict": optim.state_dict(),
                    "scheduler_dict": scheduler.state_dict(),
                    "best_loss": best_loss,
                    "loss": epoch_loss,
                    "state_dict": model.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "patience_counter": patience_counter,
                }
                torch.save(save, ckpt_file)
                save_data(config, os.path.join(output, "config.json"))
                if is_best:
                    torch.save(save, best_file)

                suffix = " [best]" if is_best else f" (patience {patience_counter}/{patience})"
                logger.info(f"  Val loss={epoch_loss:.6f}{suffix}")

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    return

        scheduler.step()

    logger.info(f"Done. Best val loss: {best_loss:.6f}")


if __name__ == "__main__":
    pretrain_combined()
