"""Contrastive + Spectral combined pretraining for SleepFM SetTransformer.

Total loss = contrastive_loss + lambda_spectral * spectral_loss

Contrastive: same leave-one-out or pairwise objective as pretrain.py.
Spectral:    SpectralDecoder predicts mean log1p band power (5 EEG bands)
             from the pooled embedding of the spectral modality (default BAS).
             Targets are computed per chunk in DataLoader workers.
"""

import datetime
import math
import os
import sys
import time

import click
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from loguru import logger
from scipy.signal import welch
from torch import nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dataset import SetTransformerDataset, collate_fn
from models.models import SetTransformer
from utils import count_parameters, load_config, load_data, save_data

def _has_nan_or_inf(state_dict):
    return any(
        torch.isnan(v).any().item() or torch.isinf(v).any().item()
        for v in state_dict.values()
        if isinstance(v, torch.Tensor)
    )


SPECTRAL_BANDS = [
    (0.5, 4.0),    # Delta
    (4.0, 8.0),    # Theta
    (8.0, 12.0),   # Alpha
    (12.0, 15.0),  # Sigma
    (15.0, 30.0),  # Beta
]


# ── Spectral decoder ─────────────────────────────────────────────────────────

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


# ── Combined dataset ──────────────────────────────────────────────────────────

class CombinedDataset(SetTransformerDataset):
    """SetTransformerDataset + per-channel spectral targets for one modality.

    __getitem__ returns:
        target_list        — list of modality tensors [C, T]
        spectral_targets   — [C_spec, 5] float32 (log1p mean band power)
        file_path, dset_names, chunk_start, modalities_length
    """

    def __init__(self, config, channel_groups, hdf5_paths=[], split="pretrain"):
        super().__init__(config, channel_groups, hdf5_paths, split)
        self.patch_size = config["patch_size"]
        self.fs = config["sampling_freq"]
        self.bands = [tuple(b) for b in config.get("spectral_bands", SPECTRAL_BANDS)]
        spectral_modality = config.get("spectral_modality", "BAS")
        modality_types = config["modality_types"]
        self.spec_idx = modality_types.index(spectral_modality) if spectral_modality in modality_types else 0

    def __getitem__(self, idx):
        target_list, file_path, dset_names, chunk_start, modalities_length = super().__getitem__(idx)
        signal = target_list[self.spec_idx].numpy().astype(np.float32)  # explicit float32
        spectral = self._compute_spectral(signal)    # [C, 5]
        return target_list, torch.from_numpy(spectral).float(), file_path, dset_names, chunk_start, modalities_length

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
                    v = float(psd[m].mean()) if m.any() else 0.0
                    bp.append(0.0 if (np.isnan(v) or np.isinf(v)) else v)
                window_powers.append(bp)
            targets[c] = np.log1p(np.mean(window_powers, axis=0))
        # Final guard in case log1p or mean produced NaN
        return np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)


def collate_fn_combined(batch):
    """Wraps collate_fn to also stack per-channel spectral targets."""
    spectral_list = [item[1] for item in batch]  # list of [C_i, 5]
    # Re-form into the standard (target_list, file, dsets, start, lengths) format
    std_batch = [(item[0], item[2], item[3], item[4], item[5]) for item in batch]
    padded_batch, mask_list, file_paths, dset_names_list, chunk_starts = collate_fn(std_batch)

    # Pad spectral targets to max channels in this batch
    max_c = max(s.shape[0] for s in spectral_list)
    B = len(batch)
    spectral_padded = torch.zeros(B, max_c, 5)
    for i, s in enumerate(spectral_list):
        spectral_padded[i, :s.shape[0]] = s

    return padded_batch, mask_list, file_paths, dset_names_list, chunk_starts, spectral_padded


# ── Combined training iteration ───────────────────────────────────────────────

def run_combined_iter(
    batch, num_modalities, model, spectral_decoder, device,
    mode, temperature, batch_size, ij,
    spectral_modality_idx, lambda_spectral,
):
    batch_data, mask_list, _, _, _, spectral_targets_batch = batch

    # Move all modality data and masks to device
    modality_data = [d.to(device, dtype=torch.float) for d in batch_data]
    modality_masks = [m.to(device, dtype=torch.bool) for m in mask_list]
    spectral_targets = spectral_targets_batch.to(device)  # [B, C_spec, 5]

    # Forward: one pass per modality
    emb_results = [model(d, m) for d, m in zip(modality_data, modality_masks)]

    # Raw (unnormalized) spectral-modality embedding for spectral loss
    raw_spec_emb = emb_results[spectral_modality_idx][0]  # [B, E]

    # Normalized embeddings for contrastive loss
    emb = [F.normalize(e[0]) for e in emb_results]

    # ── Contrastive loss ──────────────────────────────────────────────────────
    if mode == "pairwise":
        contrastive_loss = 0.0
        pairwise_loss = np.zeros((num_modalities, num_modalities), dtype=float)
        correct = np.zeros((num_modalities, num_modalities), dtype=int)
        pairs = np.zeros((num_modalities, num_modalities), dtype=int)

        for i in range(num_modalities):
            for j in range(i + 1, num_modalities):
                logits = torch.matmul(emb[i], emb[j].T) * torch.exp(temperature)
                labels = torch.arange(logits.shape[0], device=device)

                l = F.cross_entropy(logits, labels, reduction="sum")
                contrastive_loss += l
                pairwise_loss[i, j] = l.item()
                correct[i, j] = (torch.argmax(logits, dim=0) == labels).sum().item() if len(logits) else 0
                pairs[i, j] = batch_size

                l = F.cross_entropy(logits.T, labels, reduction="sum")
                contrastive_loss += l
                pairwise_loss[j, i] = l.item()
                correct[j, i] = (torch.argmax(logits, dim=1) == labels).sum().item() if len(logits) else 0
                pairs[j, i] = batch_size

        contrastive_loss /= len(ij)

    elif mode == "leave_one_out":
        contrastive_loss = 0.0
        pairwise_loss = np.zeros((num_modalities, 2), dtype=float)
        correct = np.zeros((num_modalities, 2), dtype=int)
        pairs = np.zeros((num_modalities, 2), dtype=int)

        for i in range(num_modalities):
            other = torch.stack([emb[j] for j in range(num_modalities) if j != i]).sum(0) / (num_modalities - 1)
            logits = torch.matmul(emb[i], other.T) * torch.exp(temperature)
            labels = torch.arange(logits.shape[0], device=device)

            l = F.cross_entropy(logits, labels, reduction="sum")
            contrastive_loss += l
            pairwise_loss[i, 0] = l.item()
            correct[i, 0] = (torch.argmax(logits, dim=0) == labels).sum().item() if len(logits) else 0
            pairs[i, 0] = batch_size

            l = F.cross_entropy(logits.T, labels, reduction="sum")
            contrastive_loss += l
            pairwise_loss[i, 1] = l.item()
            correct[i, 1] = (torch.argmax(logits, dim=1) == labels).sum().item() if len(logits) else 0
            pairs[i, 1] = batch_size

        contrastive_loss /= num_modalities * 2

    # ── Spectral loss ─────────────────────────────────────────────────────────
    mask_spec = modality_masks[spectral_modality_idx]  # [B, C_spec], True=padded
    valid = (~mask_spec).float().unsqueeze(-1)          # [B, C_spec, 1]
    n_real = valid.sum(dim=1).clamp(min=1)             # [B, 1]
    target_mean = (spectral_targets * valid).sum(dim=1) / n_real  # [B, 5]

    spectral_pred = spectral_decoder(raw_spec_emb)     # [B, 5]
    spectral_loss = F.mse_loss(spectral_pred, target_mean)

    total_loss = contrastive_loss + lambda_spectral * spectral_loss
    return total_loss, contrastive_loss, spectral_loss, pairwise_loss, correct, pairs


# ── Main training command ─────────────────────────────────────────────────────

@click.command("pretrain_combined")
@click.option("--config_path", type=str, default="sleepfm/configs/config_pretrain_combined.yaml")
@click.option("--channel_groups_path", type=str, default="sleepfm/configs/channel_groups.json")
@click.option("--checkpoint_path", type=str, default=None)
@click.option("--use_wandb", type=str, default=None)
def pretrain_combined(config_path, channel_groups_path, checkpoint_path, use_wandb):
    config = load_config(config_path)
    channel_groups = load_data(channel_groups_path)
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if checkpoint_path:
        output = checkpoint_path
        config = load_config(os.path.join(output, "config.json"))
    else:
        output = os.path.join(
            config["save_path"],
            f"{config['model']}/{config['mode']}_{config['embed_dim']}_patch_size_{config['patch_size']}",
        )
        os.makedirs(output, exist_ok=True)

    modality_types = config["modality_types"]
    lr = config["lr"]
    lr_step_period = config["lr_step_period"]
    gamma = config["gamma"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    temperature_val = config["temperature"]
    momentum = config["momentum"]
    num_workers = config["num_workers"]
    weight_decay = config["weight_decay"]
    mode = config["mode"]
    in_channels = config["in_channels"]
    patch_size = config["patch_size"]
    embed_dim = config["embed_dim"]
    num_heads = config["num_heads"]
    num_layers = config["num_layers"]
    pooling_head = config["pooling_head"]
    dropout = config["dropout"]
    log_interval = config["log_interval"]
    lambda_spectral = config.get("lambda_spectral", 0.5)
    spectral_modality = config.get("spectral_modality", "BAS")
    spectral_modality_idx = modality_types.index(spectral_modality)

    config["use_wandb"] = False
    os.environ["WANDB_DIR"] = output

    temperature = torch.nn.parameter.Parameter(torch.as_tensor(temperature_val))

    logger.info(f"Output: {output}")
    logger.info(f"Mode: {mode}  |  lambda_spectral: {lambda_spectral}  |  spectral_modality: {spectral_modality}")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Device: {device}")

    num_modalities = len(modality_types)
    ij = sum([((i, j), (j, i)) for i in range(num_modalities) for j in range(i + 1, num_modalities)], ())

    start = time.time()
    dataset = {
        split: CombinedDataset(config, channel_groups, split=split)
        for split in ["pretrain", "validation"]
    }
    logger.info(f"Dataset loaded in {time.time() - start:.1f}s")

    model = SetTransformer(in_channels, patch_size, embed_dim, num_heads, num_layers,
                           pooling_head=pooling_head, dropout=dropout)
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    spectral_decoder = SpectralDecoder(embed_dim, n_bands=5).to(device)

    total_layers, total_params = count_parameters(model)
    dec_params = sum(p.numel() for p in spectral_decoder.parameters() if p.requires_grad)
    logger.info(f"Encoder params: {total_params / 1e6:.2f}M  |  Decoder params: {dec_params / 1e6:.2f}M")

    optim_params = list(model.parameters()) + list(spectral_decoder.parameters())
    optim_params.append(temperature)
    optim = torch.optim.SGD(optim_params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=lr_step_period, gamma=gamma)

    epoch_resume = 0
    best_loss = math.inf

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
            model.load_state_dict(ckpt["state_dict"])
            spectral_decoder.load_state_dict(ckpt["decoder_state_dict"])
            with torch.no_grad():
                temperature.fill_(ckpt["temperature"])
            optim.load_state_dict(ckpt["optim_dict"])
            scheduler.load_state_dict(ckpt["scheduler_dict"])
            epoch_resume = ckpt["epoch"] + 1
            best_loss = ckpt["best_loss"]
            logger.info(f"Resumed from epoch {epoch_resume}, best_loss={best_loss:.6f}")
        else:
            logger.info("Starting fresh (checkpoint discarded)")
    else:
        logger.info("Starting fresh")

    os.makedirs(os.path.join(output, "log"), exist_ok=True)
    log_path = os.path.join(output, "log", f"{current_timestamp}.tsv")
    with open(log_path, "w") as log_f:
        log_f.write("Epoch\tSplit\tTotal Loss\tContrastive Loss\tSpectral Loss\tTemperature\n")
        log_f.flush()

        count_iter = 1
        for epoch in range(epoch_resume, epochs):
            split = "pretrain"
            dataloader = torch.utils.data.DataLoader(
                dataset[split], batch_size=batch_size, num_workers=num_workers,
                shuffle=True, collate_fn=collate_fn_combined, drop_last=True,
            )
            model.train()
            spectral_decoder.train()

            if mode == "pairwise":
                total_loss = total_contrastive = total_spectral = 0.0
                total_pairwise = np.zeros((num_modalities, num_modalities), dtype=float)
                total_correct = np.zeros((num_modalities, num_modalities), dtype=int)
                total_pairs = np.zeros((num_modalities, num_modalities), dtype=int)
            else:  # leave_one_out
                total_loss = total_contrastive = total_spectral = 0.0
                total_pairwise = np.zeros((num_modalities, 2), dtype=float)
                total_correct = np.zeros((num_modalities, 2), dtype=int)
                total_pairs = np.zeros((num_modalities, 2), dtype=int)
            total_n = 0

            with tqdm.tqdm(total=len(dataloader)) as pbar:
                for batch in dataloader:
                    try:
                        loss, c_loss, s_loss, pairwise, correct, pairs = run_combined_iter(
                            batch, num_modalities, model, spectral_decoder, device,
                            mode, temperature, batch_size, ij,
                            spectral_modality_idx, lambda_spectral,
                        )

                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.warning("NaN/Inf total loss — skipping batch")
                        else:
                            B = batch[0][0].size(0)
                            total_loss += loss.item()
                            total_contrastive += c_loss.item() if torch.is_tensor(c_loss) else c_loss
                            total_spectral += s_loss.item()
                            total_pairwise += pairwise
                            total_correct += correct
                            total_n += B
                            total_pairs += pairs

                            loss_per_item = loss / B
                            optim.zero_grad()
                            loss_per_item.backward()
                            optim.step()

                            if temperature < 0:
                                with torch.no_grad():
                                    temperature.fill_(0)

                            pbar.set_postfix_str(
                                f"total={total_loss/max(total_n,1):.4f} "
                                f"(c={total_contrastive/max(total_n,1):.4f} "
                                f"s={total_spectral/max(total_n,1):.4f}) "
                                f"T={temperature.item():.3f}"
                            )

                            if count_iter % config["save_iter"] == 0:
                                save = {
                                    "epoch": epoch,
                                    "temperature": temperature.item(),
                                    "optim_dict": optim.state_dict(),
                                    "scheduler_dict": scheduler.state_dict(),
                                    "best_loss": best_loss,
                                    "loss": total_loss / max(total_n, 1),
                                    "state_dict": model.state_dict(),
                                    "decoder_state_dict": spectral_decoder.state_dict(),
                                }
                                torch.save(save, ckpt_file)
                                save_data(config, os.path.join(output, "config.json"))

                            count_iter += 1

                    except Exception as exc:
                        logger.warning(f"batch error ({exc}) — skipping")

                    pbar.update()

            n = max(total_n, 1)
            log_f.write(
                f"{epoch}\t{split}\t"
                f"{total_loss/n:.6f}\t"
                f"{total_contrastive/n:.6f}\t"
                f"{total_spectral/n:.6f}\t"
                f"{temperature.item():.6f}\n"
            )
            log_f.flush()

            scheduler.step()

            epoch_loss = total_loss / n
            is_best = epoch_loss < best_loss
            if is_best:
                best_loss = epoch_loss

            save = {
                "epoch": epoch,
                "temperature": temperature.item(),
                "optim_dict": optim.state_dict(),
                "scheduler_dict": scheduler.state_dict(),
                "best_loss": best_loss,
                "loss": epoch_loss,
                "state_dict": model.state_dict(),
                "decoder_state_dict": spectral_decoder.state_dict(),
            }
            if is_best:
                torch.save(save, os.path.join(output, "best.pt"))
            torch.save(save, ckpt_file)
            save_data(config, os.path.join(output, "config.json"))

            logger.info(
                f"Epoch {epoch}: total={epoch_loss:.6f} "
                f"contrastive={total_contrastive/n:.6f} "
                f"spectral={total_spectral/n:.6f}"
                + (" [best]" if is_best else "")
            )

    logger.info("Finished Training!")


if __name__ == "__main__":
    pretrain_combined()
