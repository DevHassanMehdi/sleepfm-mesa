#!/usr/bin/env python3
"""E5: training cost (GPU-hours, wall time, epochs) for the pretraining and
fine-tuning jobs in the EEG-channel-statistics task's scope.

The account under which these jobs originally ran (hamehdi) was migrated to
the current account (hamehd) on this cluster; `sacct -u hamehdi` still
returns full historical accounting data (elapsed time, GPU count) for any
job in the window queried below -- there is no sacct retention gap once the
right account is queried. Every (job id -> run) mapping below was verified
by one of:
  - the job's own `SubmitLine` (sacct), which names the exact config file,
    model, fold, and run timestamp -- unambiguous;
  - for the two SleepFM pretraining runs, the 'epoch' field inside the
    actual encoder checkpoint (best.pt / checkpoint.pt), cross-checked
    against a per-epoch .tsv training log found inside the checkpoint
    directory whose own filename timestamp matches the job's start time;
  - for single-channel fine-tuning runs (both SleepFM and BIOT), a .log
    file recovered from /projappl/project_2019517/archive/logs_2026-11-11.tar.gz
    (extracted to a temp dir, deleted after use), matched to the job id in
    its filename.

Two groups of jobs have confirmed elapsed/GPU-hours (sacct) but no
recoverable epoch count: the SleepFM and BIOT 3-channel-BENCHMARK
fine-tuning runs. Their .log files were written under the old account's
home directory (/users/hamehdi/...), which is 0700 and not group-readable,
and predate the Nov-2026 log archive. See `missing_log_recovery_command()`
below for the exact command to run from the old account.
"""
import json
import os
import re
import subprocess

OUT_DIR = os.path.expanduser("~/analysis/eeg_channels")
ARCHIVE_LOGS = "/tmp/hamehd/logs_archive_extract/logs"  # extracted once from
# /projappl/project_2019517/archive/logs_2026-11-11.tar.gz; re-extract if missing:
#   mkdir -p /tmp/hamehd/logs_archive_extract && cd /tmp/hamehd/logs_archive_extract \
#     && tar -xzf /projappl/project_2019517/archive/logs_2026-11-11.tar.gz

ENCODERS = "/scratch/project_2019517/sleepfm-data/checkpoints/encoders"

PRETRAIN_JOBS = [
    # (model, job_id, best_pt, checkpoint_pt)
    ("SleepFM From-Scratch", 548808,
     f"{ENCODERS}/fromscratch_eegonly_fullcohort/SetTransformer/leave_one_out_128_patch_size_640/best.pt",
     f"{ENCODERS}/fromscratch_eegonly_fullcohort/SetTransformer/leave_one_out_128_patch_size_640/checkpoint.pt"),
    ("SleepFM Next-Token", 539096,
     f"{ENCODERS}/nexttoken_eegonly_fullcohort/SetTransformer/nexttoken_128_patch_size_640/best.pt",
     f"{ENCODERS}/nexttoken_eegonly_fullcohort/SetTransformer/nexttoken_128_patch_size_640/checkpoint.pt"),
]

CPU_CACHE_JOBS = [
    ("Next-Token spectral cache (CPU, prerequisite for token cache)", 539038),
    ("Next-Token token cache (CPU, k-means)", 539095),
]

EMBED_JOBS = [
    ("SleepFM From-Scratch", "EEG1 only", 2579548),
    ("SleepFM From-Scratch", "EEG2 only", 2579549),
    ("SleepFM From-Scratch", "EEG3 only", 2579550),
    ("SleepFM Next-Token", "EEG1 only", 2579551),
    ("SleepFM Next-Token", "EEG2 only", 2579552),
    ("SleepFM Next-Token", "EEG3 only", 2579553),
    ("SleepFM From-Scratch", "3-channel benchmark", 1418746),
    ("SleepFM Next-Token", "3-channel benchmark", 1390165),
]

# (model, condition) -> {fold: job_id}, fine-tuning (head training for
# SleepFM; full fine-tune for BIOT).
FINETUNE_JOBS = {
    ("SleepFM From-Scratch", "EEG1 only"): {0: 2580321, 1: 2580963, 2: 2580964, 3: 2580965, 4: 2580966},
    ("SleepFM From-Scratch", "EEG2 only"): {0: 2580967, 1: 2580968, 2: 2580969, 3: 2580970, 4: 2580971},
    ("SleepFM From-Scratch", "EEG3 only"): {0: 2580972, 1: 2580973, 2: 2580974, 3: 2580975, 4: 2580976},
    ("SleepFM From-Scratch", "3-channel, patience 10"): {0: 2613261, 1: 2613262, 2: 2613263, 3: 2613264, 4: 2613265},
    ("SleepFM From-Scratch", "3-channel benchmark"): {0: 1418927, 1: 1418928, 2: 1418929, 3: 1418930, 4: 1418931},
    ("SleepFM Next-Token", "EEG1 only"): {0: 2580322, 1: 2580977, 2: 2580978, 3: 2580979, 4: 2580980},
    ("SleepFM Next-Token", "EEG2 only"): {0: 2580981, 1: 2580982, 2: 2580983, 3: 2580984, 4: 2580985},
    ("SleepFM Next-Token", "EEG3 only"): {0: 2580986, 1: 2580987, 2: 2580988, 3: 2580989, 4: 2580990},
    ("SleepFM Next-Token", "3-channel, patience 10"): {0: 2613266, 1: 2613267, 2: 2613268, 3: 2613270, 4: 2613271},
    ("SleepFM Next-Token", "3-channel benchmark"): {0: 1390519, 1: 1390520, 2: 1390521, 3: 1390522, 4: 1390523},
    ("BIOT", "EEG1 only"): {0: 2580385, 1: 2613241, 2: 2613242, 3: 2613243, 4: 2613244},
    ("BIOT", "EEG2 only"): {0: 2613245, 1: 2613246, 2: 2613247, 3: 2613248, 4: 2613249},
    ("BIOT", "EEG3 only"): {0: 2613250, 1: 2613251, 2: 2613252, 3: 2613253, 4: 2613254},
}

# BIOT's 3-channel benchmark: each fold is one or two job segments (TIMEOUT
# then a "biot_resume" continuation), confirmed by the resume job's
# SubmitLine naming the same checkpoint dir and starting ~minutes after the
# initial job's TIMEOUT end time.
BIOT_BENCHMARK_SEGMENTS = {
    0: [539039, 561752],
    1: [539040],
    2: [539041, 561779],
    3: [539042],
    4: [539043, 561780],
}


def sacct(job_ids):
    ids = ",".join(str(j) for j in job_ids)
    out = subprocess.run(
        ["sacct", "-j", ids, "-u", "hamehdi",
         "--format=JobID,Elapsed,AllocTRES,State", "-P", "--noheader"],
        capture_output=True, text=True, timeout=30,
    ).stdout
    rows = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 4 or "." in parts[0]:
            continue
        jobid, elapsed, alloctres, state = parts[0], parts[1], parts[2], parts[3]
        h, m, s = 0, 0, 0
        days = 0
        if "-" in elapsed:
            dpart, elapsed = elapsed.split("-")
            days = int(dpart)
        bits = elapsed.split(":")
        if len(bits) == 3:
            h, m, s = (int(x) for x in bits)
        elif len(bits) == 2:
            m, s = (int(x) for x in bits)
        seconds = days * 86400 + h * 3600 + m * 60 + s
        n_gpu = 0
        gm = re.search(r"gres/gpu:[^=]+=(\d+)", alloctres)
        if gm:
            n_gpu = int(gm.group(1))
        rows[int(jobid)] = dict(elapsed_s=seconds, n_gpu=n_gpu, state=state)
    return rows


def checkpoint_epoch(path):
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    return int(d["epoch"]) if isinstance(d, dict) and "epoch" in d else None


def sleepfm_finetune_epochs(job_id):
    """Grep the archived .log for 'E<stop> Early stop | best_f1=X @ epoch <best>'."""
    import glob
    matches = glob.glob(os.path.join(ARCHIVE_LOGS, f"*_{job_id}.log"))
    if not matches:
        return None, None
    text = open(matches[0], encoding="utf-8", errors="replace").read()
    m = re.search(r"E(\d+) Early stop \| best_f1=[\d.]+ @ epoch (\d+)", text)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def biot_finetune_epochs(job_id):
    """Grep the archived .log for 'Early stop at epoch <stop> (best=X @ epoch <best>)'."""
    import glob
    matches = glob.glob(os.path.join(ARCHIVE_LOGS, f"*_{job_id}.log"))
    if not matches:
        return None, None
    text = open(matches[0], encoding="utf-8", errors="replace").read()
    m = re.search(r"Early stop at epoch (\d+) \(best=[\d.]+ @ epoch (\d+)\)", text)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def missing_log_recovery_command():
    """Jobs whose elapsed/GPU-hours are confirmed via sacct but whose epoch
    count lives in a .log file under the old account's unreadable home
    directory. Exact command to run FROM THE OLD ACCOUNT (hamehdi) to copy
    those files somewhere hamehd can read."""
    sleepfm_benchmark_jobs = []
    for cond_jobs in [FINETUNE_JOBS[("SleepFM From-Scratch", "3-channel benchmark")],
                      FINETUNE_JOBS[("SleepFM Next-Token", "3-channel benchmark")]]:
        sleepfm_benchmark_jobs.extend(cond_jobs.values())
    biot_benchmark_jobs = [j for segs in BIOT_BENCHMARK_SEGMENTS.values() for j in segs]

    sleepfm_files = " ".join(f"ft_ss_fullcohort_ft_ss_fullcohort_{j}.log" for j in sleepfm_benchmark_jobs)
    biot_initial = [BIOT_BENCHMARK_SEGMENTS[f][0] for f in range(5)]
    biot_resume = [BIOT_BENCHMARK_SEGMENTS[f][1] for f in range(5) if len(BIOT_BENCHMARK_SEGMENTS[f]) > 1]
    biot_files = (" ".join(f"biot_fullcohort_biot_fullcohort_{j}.log" for j in biot_initial) + " "
                  + " ".join(f"biot_resume_biot_resume_{j}.log" for j in biot_resume))

    dest = "/scratch/project_2019517/archive/recovered_logs_07_e5"
    return (
        f"mkdir -p {dest} && cd /users/hamehdi/projects/sleepfm-mesa/logs && "
        f"cp {sleepfm_files} {biot_files} {dest}/ && chmod -R g+r {dest}"
    )


def main():
    rows = []

    pretrain_job_ids = [j for _, j, _, _ in PRETRAIN_JOBS]
    cache_job_ids = [j for _, j in CPU_CACHE_JOBS]
    embed_job_ids = [j for _, _, j in EMBED_JOBS]
    ft_job_ids = [j for jobs in FINETUNE_JOBS.values() for j in jobs.values()]
    biot_bench_job_ids = [j for segs in BIOT_BENCHMARK_SEGMENTS.values() for j in segs]

    all_ids = pretrain_job_ids + cache_job_ids + embed_job_ids + ft_job_ids + biot_bench_job_ids
    acct = sacct(all_ids)

    for model, job_id, best_pt, ckpt_pt in PRETRAIN_JOBS:
        a = acct[job_id]
        rows.append(dict(
            step="pretrain", model=model, condition="n/a (shared across 1ch/3ch)", fold=None,
            job_ids=[job_id], elapsed_s=a["elapsed_s"], n_gpu=a["n_gpu"],
            gpu_hours=round(a["elapsed_s"] * a["n_gpu"] / 3600, 3),
            epoch_stop=checkpoint_epoch(ckpt_pt), epoch_best=checkpoint_epoch(best_pt),
            source="sacct (hamehdi) + checkpoint 'epoch' field",
        ))

    for name, job_id in CPU_CACHE_JOBS:
        a = acct[job_id]
        rows.append(dict(
            step="cpu_cache", model="SleepFM Next-Token", condition=name, fold=None,
            job_ids=[job_id], elapsed_s=a["elapsed_s"], n_gpu=a["n_gpu"],
            gpu_hours=0.0, epoch_stop=None, epoch_best=None,
            source="sacct (hamehdi)",
        ))

    for model, condition, job_id in EMBED_JOBS:
        a = acct[job_id]
        rows.append(dict(
            step="embed", model=model, condition=condition, fold=None,
            job_ids=[job_id], elapsed_s=a["elapsed_s"], n_gpu=a["n_gpu"],
            gpu_hours=round(a["elapsed_s"] * a["n_gpu"] / 3600, 3),
            epoch_stop=None, epoch_best=None,
            source="sacct (hamehdi); n/a, inference only",
        ))

    for (model, condition), fold_jobs in FINETUNE_JOBS.items():
        for fold, job_id in sorted(fold_jobs.items()):
            a = acct[job_id]
            if model == "BIOT":
                stop, best = biot_finetune_epochs(job_id)
            else:
                stop, best = sleepfm_finetune_epochs(job_id)
            source = "sacct (hamehdi) + log (archive)" if stop is not None else \
                "sacct (hamehdi); epoch count not recovered -- see recovery command"
            rows.append(dict(
                step="finetune", model=model, condition=condition, fold=fold,
                job_ids=[job_id], elapsed_s=a["elapsed_s"], n_gpu=a["n_gpu"],
                gpu_hours=round(a["elapsed_s"] * a["n_gpu"] / 3600, 3),
                epoch_stop=stop, epoch_best=best, source=source,
            ))

    for fold, segs in BIOT_BENCHMARK_SEGMENTS.items():
        total_s = sum(acct[j]["elapsed_s"] for j in segs)
        n_gpu = acct[segs[0]]["n_gpu"]
        rows.append(dict(
            step="finetune", model="BIOT", condition="3-channel benchmark", fold=fold,
            job_ids=segs, elapsed_s=total_s, n_gpu=n_gpu,
            gpu_hours=round(total_s * n_gpu / 3600, 3),
            epoch_stop=None, epoch_best=None,
            source="sacct (hamehdi); epoch count not recovered -- see recovery command",
        ))

    out = dict(rows=rows, recovery_command=missing_log_recovery_command())
    with open(os.path.join(OUT_DIR, "part_e5_training_cost.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nTotal rows: {len(rows)}")
    print(f"Total GPU-hours across all rows: {sum(r['gpu_hours'] for r in rows):.2f}")


if __name__ == "__main__":
    main()
