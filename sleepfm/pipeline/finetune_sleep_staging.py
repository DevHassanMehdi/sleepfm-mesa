import click
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger
import wandb
import yaml
import os

# Must be set before h5py is imported (via models.dataset) to take effect.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from datetime import datetime
import sys

SLEEPFM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SLEEPFM_DIR)
sys.path.append(SLEEPFM_DIR)

from utils import *
from models.models import SleepEventLSTMClassifier
from models.dataset import SleepEventClassificationDataset as Dataset
from models.dataset import sleep_event_finetune_full_collate_fn as collate_fn
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score


def masked_cross_entropy_loss(outputs, y_data, mask, device):
    # Reshape outputs and labels to (B * seq_len, num_classes) and (B * seq_len,)
    B, seq_len, num_classes = outputs.shape
    outputs = outputs.reshape(B * seq_len, num_classes)
    y_data = y_data.reshape(B * seq_len).long()  # Convert y_data to Long for cross_entropy
    mask = mask.reshape(B * seq_len)

    # Inverse-frequency class weights computed from actual label distribution
    # Wake: 20.3% (14013), N1: 30.1% (20783), N2: 34.4% (23720), N3: 9.4% (6518), REM: 5.8% (4008)
    counts = torch.tensor([14013, 20783, 23720, 6518, 4008], dtype=torch.float32, device=device)
    total = counts.sum()
    weights_tensor = total / (num_classes * counts)
    # Normalize so mean weight = 1.0
    weights_tensor = weights_tensor / weights_tensor.mean()

    loss = F.cross_entropy(outputs, y_data, weight=weights_tensor, reduction='none')

    loss = loss * (mask == 0).float()

    loss = loss.sum() / (mask == 0).float().sum()

    return loss


@click.command("finetune_sleep_staging")
@click.option("--config_path", type=str, default=os.path.join(SLEEPFM_DIR, "configs/config_finetune_sleep_events.yaml"))
@click.option("--channel_groups_path", type=str, default=os.path.join(SLEEPFM_DIR, "configs/channel_groups.json"))
@click.option("--checkpoint_path", type=str, default=None)
@click.option("--split_path", type=str, default=None)
@click.option("--train_split", type=str, default="train")
@click.option("--fold", type=int, default=0)
def finetune_sleep_staging(config_path, channel_groups_path, checkpoint_path, split_path, train_split, fold):
    # Load configuration
    config = load_config(config_path)
    channel_groups = load_config(channel_groups_path)

    # Resolve relative config paths against the repo root so the script
    # works regardless of the current working directory.
    for key in ["data_path", "model_path", "split_path", "labels_path"]:
        if config.get(key) and not os.path.isabs(config[key]):
            config[key] = os.path.join(REPO_ROOT, config[key])

    prefix = config["labels_path"].split("/")[-1]
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if split_path:
        if not os.path.isabs(split_path):
            split_path = os.path.join(REPO_ROOT, split_path)
        config["split_path"] = split_path

    # Select the requested fold from the 10-fold split and write it out as a
    # flat {train, validation, test} split for the dataset class to consume.
    full_split = load_data(config["split_path"])
    fold_split = full_split[f"fold_{fold}"]
    fold_split_path = os.path.join(REPO_ROOT, "data/mesa", f"dataset_split_10fold_fold{fold}.json")
    save_data(fold_split, fold_split_path)
    config["split_path"] = fold_split_path

    split_path = config["split_path"]
    channel_like = config["channel_like"]
    channel_like_string = "_".join(channel_like)

    dataset_prefix = "_".join(config["dataset"].split(","))

    if checkpoint_path:
        output = checkpoint_path
        config = load_data(os.path.join(output, "config.json"))
    else:
        scratch_base = "/scratch/project_2019517/sleepfm-data/checkpoints"
        output = os.path.join(scratch_base, f"{config['model']}_{dataset_prefix}_{prefix}_{channel_like_string}", f"fold_{fold}")
        os.makedirs(output, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Initialize model
    model_params = config['model_params']
    model_class = getattr(sys.modules[__name__], config['model'])
    model = model_class(**model_params).to(device)
    model_name = type(model).__name__

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    total_layers, total_params = count_parameters(model)
    logger.info(f"Device: {device.type} | Model: {model_name} | Params: {total_params / 1e6:.2f}M")

    # Initialize dataset and dataloaders
    batch_size = config.get('batch_size', 1)
    num_workers = config.get('num_workers', 4)

    train_dataset = Dataset(config, channel_groups, split=train_split)
    val_dataset = Dataset(config, channel_groups, split="validation")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    logger.info(f"Data: fold={fold} | train={len(train_dataset)} | val={len(val_dataset)}")

    # Optimizer and loss function
    num_epochs = config.get('epochs', 500)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    start_epoch = 0
    if checkpoint_path:
        checkpoint_path = os.path.join(output, "checkpoint.pth")
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Set up Weights & Biases
    if config["use_wandb"]:
        current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        wandb.init(project="PSG-fm", name=f"run_at_{current_timestamp}", config=config)

    # Training loop
    best_val_f1 = -float('inf')
    best_epoch = 0
    patience_counter = 0
    patience = 50

    class_abbrev = ['W', 'N1', 'N2', 'N3', 'R']

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        for x_data, y_data, padded_matrix, hdf5_path_list in train_loader:
            x_data, y_data, padded_matrix = x_data.to(device), y_data.to(device), padded_matrix.to(device)
            outputs, mask = model(x_data, padded_matrix)
            loss = masked_cross_entropy_loss(outputs, y_data, mask, device)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # Validation loop at the end of each epoch
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        all_masks = []
        with torch.no_grad():
            for x_data, y_data, padded_matrix, _ in val_loader:
                x_data, y_data, padded_matrix = x_data.to(device), y_data.to(device), padded_matrix.to(device)
                outputs, mask = model(x_data, padded_matrix)
                loss = masked_cross_entropy_loss(outputs, y_data, mask, device)
                val_loss += loss.item()
                all_preds.append(outputs.argmax(dim=2).cpu().numpy())
                all_targets.append(y_data.cpu().numpy())
                all_masks.append(mask.cpu().numpy())

        val_loss /= len(val_loader)

        # Compute F1 scores on valid (unmasked) predictions
        all_preds_flat = np.concatenate([p.flatten() for p in all_preds])
        all_targets_flat = np.concatenate([t.flatten() for t in all_targets])
        all_masks_flat = np.concatenate([m.flatten() for m in all_masks])
        valid_mask = all_masks_flat == 0
        all_preds_valid = all_preds_flat[valid_mask]
        all_targets_valid = all_targets_flat[valid_mask]

        val_f1 = f1_score(all_targets_valid, all_preds_valid, average='macro', zero_division=0)
        per_class_f1 = f1_score(all_targets_valid, all_preds_valid, average=None, zero_division=0)

        # Format per-class F1 scores
        per_class_str = " ".join([f"{abbr}={f1:.2f}" for abbr, f1 in zip(class_abbrev, per_class_f1)])

        # Plain text epoch log
        best_marker = " *" if val_f1 > best_val_f1 else ""
        epoch_log = f"E{epoch + 1:03d} loss={train_loss:.3f} vl={val_loss:.3f} vf1={val_f1:.3f} | {per_class_str}{best_marker}"
        logger.info(epoch_log)

        # Log to wandb if enabled
        if config["use_wandb"]:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1": val_f1,
                "epoch": epoch + 1
            })

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_path = os.path.join(output, "best.pth")
            torch.save(model.state_dict(), best_model_path)
            save_data(config, os.path.join(output, "config.json"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"E{epoch + 1} Early stop | best_f1={best_val_f1:.3f} @ epoch {best_epoch}")
                break

        scheduler.step()
        model.train()

if __name__ == "__main__":
    finetune_sleep_staging()