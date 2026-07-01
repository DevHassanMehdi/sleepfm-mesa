"""
Evaluate a fine-tuned LaBraM checkpoint on the MESA held-out test split.

Usage:
    python scripts/evaluate_labram.py --modality EEG_ONLY
"""
import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, "/scratch/project_2019517/LaBraM")

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from labram_dataset import LaBraMSleepDataset, MODALITY_CHANNELS, get_ch_names
from finetune_labram import build_model, forward_logits, get_input_chans, CKPT_ROOT, SPLIT_PATH

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", required=True, choices=list(MODALITY_CHANNELS.keys()))
    parser.add_argument("--fold_key", default="fold_0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = os.path.join(CKPT_ROOT, args.modality, args.fold_key, "best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    test_ds = LaBraMSleepDataset(SPLIT_PATH, "test", args.modality, fold_key=args.fold_key)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    print(f"[{args.modality}] test={len(test_ds)}", flush=True)

    model = build_model(args.modality).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    input_chans = torch.tensor(get_input_chans(get_ch_names(args.modality)), device=device)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating"):
            x = x.to(device)
            logits = forward_logits(model, x, input_chans)
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_targets.append(y.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    acc = (all_preds == all_targets).mean()
    report = classification_report(
        all_targets, all_preds, labels=[0, 1, 2, 3, 4],
        target_names=STAGE_NAMES, zero_division=0
    )

    lines = [
        f"LaBraM FINE-TUNED ({args.modality}) — MESA held-out test split",
        "=" * 50,
        f"Channels: {', '.join(MODALITY_CHANNELS[args.modality])} "
        f"-> {', '.join(get_ch_names(args.modality))} (10-20 mapping, see labram_dataset.py)",
        "Pretrained: labram-base.pth (full fine-tune, encoder unfrozen)",
        "",
        f"Macro F1:  {macro_f1:.4f}",
        f"Accuracy:  {acc:.4f}",
        "",
        report,
    ]
    result_text = "\n".join(lines)
    print("\n" + result_text)

    out_path = os.path.join(REPO_ROOT, "results", f"labram_{args.modality}_results.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(result_text)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
