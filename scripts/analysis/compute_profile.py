"""Compute profile for results/07 (Part E): parameters, FLOPs, input data
size, and (optionally) latency/throughput/memory benchmarking.

Usage:
    python scripts/analysis/compute_profile.py --params --flops --input-size
    python scripts/analysis/compute_profile.py --benchmark   # needs a GPU

Each flag is independent; results are printed and written as JSON under
~/analysis/eeg_channels/ (never under git or a checkpoint folder).
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENC_ROOT = "/scratch/project_2019517/sleepfm-data/checkpoints/encoders"
CKPT = f"{REPO_ROOT}/data/checkpoints/full_cohort"
OUT_DIR = os.path.expanduser("~/analysis/eeg_channels")

sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "sleepfm"))
sys.path.insert(0, "/scratch/project_2019517/BIOT")


# ---------------------------------------------------------------------------
# E1: parameters, checkpoint size, fp16 estimate.
# ---------------------------------------------------------------------------

def count_params(path):
    """Total parameter count, on-disk file size, and weights-ONLY size
    (sum of just the state_dict tensors' bytes) of a .pt/.pth checkpoint.
    Handles the three wrapping conventions seen in this project: a bare
    state_dict (BIOT, and SleepFM's fine-tuned head), {"model_state_dict":
    ...} (BIOT/LaBraM/SensorLM eval scripts), or {"state_dict": ...}
    (SleepFM encoder pretraining checkpoints -- see run_params()'s note on
    what else these additionally carry)."""
    import torch
    raw = torch.load(path, map_location="cpu", weights_only=False)
    sd = raw
    extra_keys = []
    if isinstance(sd, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in sd and isinstance(sd[key], dict):
                extra_keys = [k for k in sd.keys() if k != key]
                sd = sd[key]
                break
    total = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    weights_only_bytes = sum(v.numel() * v.element_size() for v in sd.values() if hasattr(v, "numel"))
    size_bytes = os.path.getsize(path)
    return total, size_bytes, weights_only_bytes, extra_keys


def run_params():
    """SleepFM fine-tuning uses precomputed embeddings: the encoder
    (pretraining checkpoint, frozen at fine-tuning time) and the head
    (SleepEventLSTMClassifier, the only trainable part) are reported
    separately and summed. BIOT is a single end-to-end checkpoint, fully
    fine-tuned (no freezing in scripts/finetune_biot.py)."""
    encoders = {
        ("SleepFM From-Scratch", "1 channel"): f"{ENC_ROOT}/fromscratch_eegonly_fullcohort_EEG1_ONLY/SetTransformer/leave_one_out_128_patch_size_640/best.pt",
        ("SleepFM From-Scratch", "3 channels"): f"{ENC_ROOT}/fromscratch_eegonly_fullcohort/SetTransformer/leave_one_out_128_patch_size_640/best.pt",
        ("SleepFM Next-Token", "1 channel"): f"{ENC_ROOT}/nexttoken_eegonly_fullcohort_EEG1_ONLY/SetTransformer/nexttoken_128_patch_size_640/best.pt",
        ("SleepFM Next-Token", "3 channels"): f"{ENC_ROOT}/nexttoken_eegonly_fullcohort/SetTransformer/nexttoken_128_patch_size_640/best.pt",
    }
    heads = {
        ("SleepFM From-Scratch", "1 channel"): f"{CKPT}/sleepfm__EEG1_ONLY__fromscratch__fold5_v1__2026-10-27_1521/fold_0/best.pth",
        ("SleepFM From-Scratch", "3 channels"): f"{CKPT}/sleepfm__EEG_ONLY__fromscratch__fold5_v1__2026-09-19_1657/fold_0/best.pth",
        ("SleepFM Next-Token", "1 channel"): f"{CKPT}/sleepfm__EEG1_ONLY__nexttoken__fold5_v1__2026-10-27_1522/fold_0/best.pth",
        ("SleepFM Next-Token", "3 channels"): f"{CKPT}/sleepfm__EEG_ONLY__nexttoken__fold5_v1__2026-09-17_2326/fold_0/best.pth",
    }
    biot = {
        ("BIOT", "1 channel"): f"{CKPT}/biot__EEG1_ONLY__finetuned__fold5_v1__2026-10-27_1523/fold_0/best.pth",
        ("BIOT", "3 channels"): f"{CKPT}/biot__EEG_ONLY__finetuned__fold5_v1__2026-08-09_1229/fold_0/best.pth",
    }

    rows = []
    for key, enc_path in encoders.items():
        model, cond = key
        enc_params, enc_size, enc_weights_only, enc_extra = count_params(enc_path)
        head_params, head_size, head_weights_only, head_extra = count_params(heads[key])
        extra_note = ("From-Scratch encoder checkpoint also stores optimizer (Adam) state and LR "
                      "scheduler state alongside the weights" if "From-Scratch" in model else
                      "Next-Token encoder checkpoint also stores optimizer (Adam) state, LR scheduler "
                      "state, AND a small next-token-prediction pretext head (head_state_dict, ~0.25MB, "
                      "not used at fine-tuning time) alongside the weights")
        rows.append(dict(
            model=model, condition=cond,
            encoder_params=enc_params, head_params=head_params, trainable_params=head_params,
            total_params=enc_params + head_params,
            ckpt_size_mb=(enc_size + head_size) / 1024 / 1024,
            weights_only_mb=(enc_weights_only + head_weights_only) / 1024 / 1024,
            fp16_estimate_mb=(enc_params + head_params) * 2 / 1024 / 1024,
            note="encoder is frozen during fine-tuning; only the head is trainable. " + extra_note,
        ))
    for key, path in biot.items():
        model, cond = key
        total, size, weights_only, extra = count_params(path)
        rows.append(dict(
            model=model, condition=cond,
            encoder_params=None, head_params=None, trainable_params=total,
            total_params=total, ckpt_size_mb=size / 1024 / 1024,
            weights_only_mb=weights_only / 1024 / 1024,
            fp16_estimate_mb=total * 2 / 1024 / 1024,
            note="single end-to-end checkpoint, fully fine-tuned, bare state_dict (no optimizer/scheduler state saved)",
        ))
    return rows


# ---------------------------------------------------------------------------
# E2: FLOPs per 30s epoch and per 8h night.
# ---------------------------------------------------------------------------

EPOCHS_PER_NIGHT = 960  # 8h / 30s


def run_flops():
    """torch.utils.flop_counter.FlopCounterMode (built into PyTorch, no
    extra dependency): hooks aten ops via __torch_dispatch__ and counts
    matrix multiplies and convolutions (mm/bmm/addmm/conv*d, and the
    matmuls inside scaled-dot-product attention) -- not elementwise ops
    (activations, softmax, normalization, residual adds) and not BIOT's
    STFT front end (torch.stft is not a matmul primitive). Those are a
    small fraction of the total at these model sizes and are excluded
    below. Shape/architecture-determined, run on CPU or GPU equally."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    def count(fn):
        with FlopCounterMode(display=False) as fc:
            fn()
        return fc.get_total_flops()

    rows = []
    from finetune_biot import build_model as biot_build_model
    for n_ch, label in [(1, "1 channel"), (3, "3 channels")]:
        model = biot_build_model().eval()
        x = torch.randn(1, n_ch, 6000)
        with torch.no_grad():
            flops_epoch = count(lambda: model(x))
        rows.append(dict(model="BIOT", component="end-to-end", condition=label,
                          flops_per_epoch=flops_epoch, flops_per_night=flops_epoch * EPOCHS_PER_NIGHT))

    from models.models import SetTransformer, SleepEventLSTMClassifier
    EMBED_DIM = 128
    for n_ch, label in [(1, "1 channel"), (3, "3 channels")]:
        encoder = SetTransformer(in_channels=1, embed_dim=EMBED_DIM, num_heads=8, num_layers=6,
                                  pooling_head=8, dropout=0.0, patch_size=640).eval()
        x = torch.randn(1, n_ch, 3840)
        mask = torch.zeros(1, n_ch, dtype=torch.bool)
        with torch.no_grad():
            flops_epoch = count(lambda: encoder(x, mask))
        rows.append(dict(model="SleepFM (From-Scratch/Next-Token share this encoder architecture)",
                          component="encoder", condition=label,
                          flops_per_epoch=flops_epoch, flops_per_night=flops_epoch * EPOCHS_PER_NIGHT))

    head = SleepEventLSTMClassifier(embed_dim=EMBED_DIM, num_heads=4, num_layers=2, num_classes=5,
                                     pooling_head=4, dropout=0.0, max_seq_length=EPOCHS_PER_NIGHT).eval()
    n_ch_for_head = 3
    emb_seq = torch.randn(1, n_ch_for_head, EPOCHS_PER_NIGHT, EMBED_DIM)
    mask_head = torch.zeros(1, n_ch_for_head, EPOCHS_PER_NIGHT, dtype=torch.bool)
    with torch.no_grad():
        flops_night = count(lambda: head(emb_seq, mask_head, mask_head))
    rows.append(dict(model="SleepFM (From-Scratch/Next-Token share this architecture)",
                      component="head", condition="n/a (operates on embeddings, whole night at once)",
                      flops_per_epoch=flops_night / EPOCHS_PER_NIGHT, flops_per_night=flops_night))

    # "Together" = encoder run once per epoch across the night + head run
    # once per night, for each channel condition (head input channel
    # count fixed at the 3-channel case it was actually trained on).
    by_key = {(r["model"], r["condition"]): r for r in rows if r["component"] == "encoder"}
    head_row = next(r for r in rows if r["component"] == "head")
    for (model, cond), enc_row in by_key.items():
        rows.append(dict(model=model, component="encoder+head together", condition=cond,
                          flops_per_epoch=enc_row["flops_per_epoch"] + head_row["flops_per_epoch"],
                          flops_per_night=enc_row["flops_per_night"] + head_row["flops_per_night"]))
    return rows


# ---------------------------------------------------------------------------
# E4: input data size per channel per night.
# ---------------------------------------------------------------------------

def run_input_size():
    """MESA EEG is stored at 128 Hz (sleepfm/experiment_paths.py,
    biot_dataset.py MESA_HZ). SleepFM's SetTransformer consumes that
    native 128 Hz signal directly. BIOT resamples each 30s epoch to
    200 Hz before its STFT front end (biot_dataset.py BIOT_HZ). An
    8h night = 960 30-second epochs = 28800 s."""
    NIGHT_SECONDS = 8 * 3600
    BYTES_PER_SAMPLE = 4  # float32
    rows = []
    for model, hz in [("SleepFM From-Scratch", 128), ("SleepFM Next-Token", 128), ("BIOT", 200)]:
        samples = hz * NIGHT_SECONDS
        rows.append(dict(model=model, sampling_rate_hz=hz,
                          samples_per_channel_per_night=samples,
                          bytes_per_channel_per_night=samples * BYTES_PER_SAMPLE,
                          mb_per_channel_per_night=samples * BYTES_PER_SAMPLE / 1024 / 1024))
    return rows


# ---------------------------------------------------------------------------
# E3 (optional, needs a GPU): latency/throughput/memory. See the module
# docstring in part_e3_latency.py for the full methodology; this is the
# same logic, callable via --benchmark.
# ---------------------------------------------------------------------------

def run_benchmark(node_label="shared-node", n_warmup=10, n_timed=100, n_repeats=3, device_scope="both"):
    """Benchmarks each model with ITS OWN per-condition input shape (BIOT:
    (B, n_ch, 6000); SleepFM encoder: (B, n_ch, 3840) with n_ch in {1, 3}),
    plus the SleepFM staging head on embeddings (n_ch=3, the shape it is
    actually trained/used with -- see run_flops()'s head FLOPs note).
    Each (model, condition, device) cell is measured n_repeats independent
    times, each with its own n_warmup warm-up calls and n_timed timed calls,
    CUDA-synchronized around every call. Excludes data loading and
    preprocessing -- only the model forward pass is timed."""
    import numpy as np
    import torch

    def _run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            return f"(unavailable on this node: {e})"

    def gpu_memory_record():
        return _run(["nvidia-smi", "--query-gpu=index,utilization.gpu,utilization.memory,"
                     "memory.used,memory.total", "--format=csv"])

    def node_squeue_record():
        node = socket.gethostname()
        return node, _run(["squeue", "-w", node, "--format=%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"])

    def hardware_record():
        cpu = _run(["lscpu"])
        cpu_model = next((l.split(":", 1)[1].strip() for l in cpu.splitlines() if l.startswith("Model name:")), "unknown")
        cpu_count = next((l.split(":", 1)[1].strip() for l in cpu.splitlines() if l.strip().startswith("CPU(s):")), "unknown")
        gpu_name = _run(["nvidia-smi", "-L"]).strip()
        gpu_driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).strip()
        return dict(cpu_model=cpu_model, cpu_count=cpu_count,
                    gpu_name=gpu_name or "n/a (no GPU on this node)",
                    gpu_driver=gpu_driver or "n/a (no GPU on this node)")

    def timed_runs(fn, device):
        for _ in range(n_warmup):
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
        latencies = []
        for _ in range(n_timed):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
        return np.array(latencies)

    def repeated_cell(fn, device, batch_size, peak_mem_device=None):
        """Runs n_repeats independent (warmup + timed) measurements and
        returns per-repeat median/p95 lists plus one peak-memory reading
        (memory doesn't need repeating -- it's deterministic per shape)."""
        medians, p95s, throughputs = [], [], []
        peak_mb = None
        for rep in range(n_repeats):
            if peak_mem_device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            lat = timed_runs(fn, device)
            medians.append(float(np.median(lat)))
            p95s.append(float(np.percentile(lat, 95)))
            throughputs.append(batch_size / float(np.median(lat)))
            if peak_mem_device == "cuda" and rep == n_repeats - 1:
                peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        out = dict(median_s_per_repeat=medians, p95_s_per_repeat=p95s,
                    throughput_epochs_per_s_per_repeat=throughputs)
        if peak_mb is not None:
            out["peak_gpu_memory_mb"] = peak_mb
        return out

    device = "cuda" if torch.cuda.is_available() else "cpu"
    node_name, squeue_before = node_squeue_record()
    gpu_before = gpu_memory_record()
    hardware = hardware_record()

    from finetune_biot import build_model as biot_build_model
    from models.models import SetTransformer, SleepEventLSTMClassifier

    rows = []

    want_gpu = device_scope in ("both", "gpu") and device == "cuda"
    want_cpu = device_scope in ("both", "cpu")

    # --- BIOT: end-to-end, per channel-count condition ---
    for n_ch, label in [(1, "1 channel"), (3, "3 channels")]:
        if want_gpu:
            biot_model = biot_build_model().to(device).eval()
            for b in [1, 16, 128]:
                x = torch.randn(b, n_ch, 6000, device=device)
                with torch.no_grad():
                    cell = repeated_cell(lambda: biot_model(x), device, b,
                                          peak_mem_device="cuda" if b == 128 else None)
                cell.update(model="BIOT", component="end-to-end", condition=label,
                            device=f"GPU batch={b}", node_label=node_label)
                rows.append(cell)

        if want_cpu:
            biot_cpu = biot_build_model().to("cpu").eval()
            for n_threads in [1, 8]:
                torch.set_num_threads(n_threads)
                x = torch.randn(16, n_ch, 6000)
                with torch.no_grad():
                    cell = repeated_cell(lambda: biot_cpu(x), "cpu", 16)
                cell.update(model="BIOT", component="end-to-end", condition=label,
                            device=f"CPU threads={n_threads}", node_label=node_label)
                rows.append(cell)

    # --- SleepFM encoder, per channel-count condition. CORRECTED shape:
    # one forward call processes a 5-MINUTE chunk (38400 samples @ 128Hz --
    # sampling_duration=5(min) in every pretraining config.json,
    # sampling_freq=128 -- sleepfm/models/dataset.py's
    # SetTransformerDataset.samples_per_chunk), not a 30s/3840-sample
    # epoch as previously (wrongly) benchmarked. See results/07 Part E2's
    # note for the full evidence trail. ---
    SAMPLES_PER_CHUNK = 5 * 60 * 128  # 38400
    for n_ch, label in [(1, "1 channel"), (3, "3 channels")]:
        if want_gpu:
            encoder = SetTransformer(in_channels=1, embed_dim=128, num_heads=8, num_layers=6,
                                      pooling_head=8, dropout=0.0, patch_size=640).to(device).eval()
            for b in [1, 16, 128]:
                x = torch.randn(b, n_ch, SAMPLES_PER_CHUNK, device=device)
                mask = torch.zeros(b, n_ch, dtype=torch.bool, device=device)
                with torch.no_grad():
                    cell = repeated_cell(lambda: encoder(x, mask), device, b,
                                          peak_mem_device="cuda" if b == 128 else None)
                cell.update(model="SleepFM (From-Scratch/Next-Token share this encoder architecture)",
                            component="encoder", condition=label, device=f"GPU batch={b}", node_label=node_label,
                            real_shape=f"(B, {n_ch}, {SAMPLES_PER_CHUNK}) -- 5-min chunk")
                rows.append(cell)

        if want_cpu:
            encoder_cpu = SetTransformer(in_channels=1, embed_dim=128, num_heads=8, num_layers=6,
                                          pooling_head=8, dropout=0.0, patch_size=640).to("cpu").eval()
            for n_threads in [1, 8]:
                torch.set_num_threads(n_threads)
                x = torch.randn(16, n_ch, SAMPLES_PER_CHUNK)
                mask = torch.zeros(16, n_ch, dtype=torch.bool)
                with torch.no_grad():
                    cell = repeated_cell(lambda: encoder_cpu(x, mask), "cpu", 16)
                cell.update(model="SleepFM (From-Scratch/Next-Token share this encoder architecture)",
                            component="encoder", condition=label, device=f"CPU threads={n_threads}", node_label=node_label,
                            real_shape=f"(16, {n_ch}, {SAMPLES_PER_CHUNK}) -- 5-min chunk")
                rows.append(cell)

    # --- SleepFM staging head, on embeddings (n_ch=3, as actually trained -- see run_flops()) ---
    EPOCHS_PER_NIGHT = 960
    n_ch_head = 3
    if want_gpu:
        head = SleepEventLSTMClassifier(embed_dim=128, num_heads=4, num_layers=2, num_classes=5,
                                         pooling_head=4, dropout=0.0, max_seq_length=EPOCHS_PER_NIGHT).to(device).eval()
        for b in [1, 16, 128]:
            emb = torch.randn(b, n_ch_head, EPOCHS_PER_NIGHT, 128, device=device)
            m = torch.zeros(b, n_ch_head, EPOCHS_PER_NIGHT, dtype=torch.bool, device=device)
            with torch.no_grad():
                cell = repeated_cell(lambda: head(emb, m, m), device, b,
                                      peak_mem_device="cuda" if b == 128 else None)
            cell.update(model="SleepFM (From-Scratch/Next-Token share this architecture)",
                        component="head (whole night, n_ch=3)", condition="n/a (operates on embeddings)",
                        device=f"GPU batch={b}", node_label=node_label)
            rows.append(cell)

    if want_cpu:
        head_cpu = SleepEventLSTMClassifier(embed_dim=128, num_heads=4, num_layers=2, num_classes=5,
                                             pooling_head=4, dropout=0.0, max_seq_length=EPOCHS_PER_NIGHT).to("cpu").eval()
        for n_threads in [1, 8]:
            torch.set_num_threads(n_threads)
            emb = torch.randn(16, n_ch_head, EPOCHS_PER_NIGHT, 128)
            m = torch.zeros(16, n_ch_head, EPOCHS_PER_NIGHT, dtype=torch.bool)
            with torch.no_grad():
                cell = repeated_cell(lambda: head_cpu(emb, m, m), "cpu", 16)
            cell.update(model="SleepFM (From-Scratch/Next-Token share this architecture)",
                        component="head (whole night, n_ch=3)", condition="n/a (operates on embeddings)",
                        device=f"CPU threads={n_threads}", node_label=node_label)
            rows.append(cell)

    gpu_after = gpu_memory_record()
    _, squeue_after = node_squeue_record()
    return dict(node=node_name, node_label=node_label, n_repeats=n_repeats, hardware=hardware,
                squeue_before=squeue_before, squeue_after=squeue_after,
                gpu_before=gpu_before, gpu_after=gpu_after, results=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", action="store_true")
    ap.add_argument("--flops", action="store_true")
    ap.add_argument("--input-size", action="store_true")
    ap.add_argument("--benchmark", action="store_true", help="needs a GPU")
    ap.add_argument("--node-label", default="shared-node",
                     help="'shared-node' (default) or 'exclusive-node', recorded in the output")
    ap.add_argument("--n-repeats", type=int, default=3)
    ap.add_argument("--n-warmup", type=int, default=10)
    ap.add_argument("--n-timed", type=int, default=100)
    ap.add_argument("--device-scope", default="both", choices=["both", "gpu", "cpu"],
                     help="restrict --benchmark to just the GPU or just the CPU measurements "
                          "(for running each on its own dedicated exclusive node)")
    args = ap.parse_args()
    if not any([args.params, args.flops, args.input_size, args.benchmark]):
        args.params = args.flops = args.input_size = True

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.params:
        rows = run_params()
        print(json.dumps(rows, indent=2))
        with open(os.path.join(OUT_DIR, "part_e1_params.json"), "w") as f:
            json.dump(rows, f, indent=2)

    if args.flops:
        rows = run_flops()
        print(json.dumps(rows, indent=2))
        with open(os.path.join(OUT_DIR, "part_e2_flops.json"), "w") as f:
            json.dump(rows, f, indent=2)

    if args.input_size:
        rows = run_input_size()
        print(json.dumps(rows, indent=2))
        with open(os.path.join(OUT_DIR, "part_e4_input_size.json"), "w") as f:
            json.dump(rows, f, indent=2)

    if args.benchmark:
        data = run_benchmark(node_label=args.node_label, n_warmup=args.n_warmup,
                              n_timed=args.n_timed, n_repeats=args.n_repeats,
                              device_scope=args.device_scope)
        print(json.dumps(data, indent=2))
        suffix = "shared" if args.node_label == "shared-node" else "exclusive"
        scope_tag = "" if args.device_scope == "both" else f"_{args.device_scope}"
        with open(os.path.join(OUT_DIR, f"part_e3_latency_{suffix}{scope_tag}.json"), "w") as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
