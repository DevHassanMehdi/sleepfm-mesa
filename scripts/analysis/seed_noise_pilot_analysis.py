"""Part 1 of the seed-noise follow-up task: pure analysis (no GPU) of the
30 existing pilot runs (SleepFM From-Scratch/Next-Token, EEG1/EEG2/EEG3,
fold 0, seeds 1-5, patience 10) and their SLURM logs. Read-only -- writes
nothing under results/ or into any checkpoint folder.

Reusable, testable helpers live here; the __main__ block at the bottom
loads the real pilot data and prints every Part-1 table.
"""
import glob
import os
import re
import statistics as st

import numpy as np
from scipy import stats as scipy_stats

EPOCH_LINE_RE = re.compile(r"E(\d+) loss=[\d.]+ vl=[\d.]+ vf1=([\d.]+)")
EARLY_STOP_RE = re.compile(r"E(\d+) Early stop \| best_f1=([\d.]+) @ epoch (\d+)")
RUN_SEED_RE = re.compile(r"^RUN:.*finetune_sleep_staging\.py.*--seed (\d+)", re.MULTILINE)
RUN_LINE_RE = re.compile(r"^RUN:", re.MULTILINE)


def parse_stop_and_best_epoch(segment_text):
    """Authoritative stop_epoch/best_epoch straight from the script's own
    "E<n> Early stop | best_f1=<f> @ epoch <m>" log line -- independent of
    whether the surrounding per-epoch trajectory got truncated."""
    m = EARLY_STOP_RE.search(segment_text)
    if not m:
        return None, None, None
    stop_epoch, best_f1, best_epoch = int(m.group(1)), float(m.group(2)), int(m.group(3))
    return stop_epoch, best_epoch, best_f1


def split_log_into_seed_segments(log_text):
    """Split one pilot SLURM log (5 sequential seeds) into per-seed text
    segments, keyed by seed number. Each segment runs from a
    finetune_sleep_staging.py "RUN:" line to the next "RUN:" line (the
    matching evaluate_sleep_staging.py call) -- it's the truncated
    (last ~5000 chars of each subprocess's merged stdout/stderr) segment
    the driver script printed, which keeps the TAIL of the run (ending at
    the stop), not the head.
    """
    run_line_starts = [m.start() for m in RUN_LINE_RE.finditer(log_text)]
    seed_starts = [(m.start(), int(m.group(1))) for m in RUN_SEED_RE.finditer(log_text)]
    segments = {}
    for start, seed in seed_starts:
        later_run_lines = [s for s in run_line_starts if s > start]
        end = min(later_run_lines) if later_run_lines else len(log_text)
        segments[seed] = log_text[start:end]
    return segments


def parse_epoch_trajectory(segment_text):
    """Return [(epoch:int, val_f1:float), ...] in order, for one seed's
    (possibly truncated-at-the-front) log segment."""
    return [(int(e), float(f1)) for e, f1 in EPOCH_LINE_RE.findall(segment_text)]


def flatness_last_n_epochs(trajectory, n=30):
    """SD and range of val F1 over the last min(n, len(trajectory)) epochs
    of a run's trajectory (trajectory assumed already ending at the stop
    epoch). Returns (n_used, sd, rng)."""
    if not trajectory:
        return 0, float("nan"), float("nan")
    tail = trajectory[-n:]
    vals = [v for _, v in tail]
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    rng = max(vals) - min(vals)
    return len(tail), sd, rng


def chi2_sd_ci(values, confidence=0.95):
    """CI for a sample standard deviation via the chi-square distribution
    on (n-1)*s^2/sigma^2 ~ chi2(n-1). Returns (sd, ci_lo, ci_hi)."""
    n = len(values)
    s = st.stdev(values)
    df = n - 1
    alpha = 1 - confidence
    chi2_hi = scipy_stats.chi2.ppf(1 - alpha / 2, df)
    chi2_lo = scipy_stats.chi2.ppf(alpha / 2, df)
    ci_lo = s * np.sqrt(df / chi2_hi)
    ci_hi = s * np.sqrt(df / chi2_lo)
    return s, ci_lo, ci_hi


def expected_fold_gap_sd(seed_sd, n_folds=5):
    """Expected SD of the difference between two conditions' fold-mean
    macro F1 (each mean over n_folds independent folds), UNDER THE
    ASSUMPTION that fold-to-fold variance equals the measured seed-to-seed
    variance at fixed fold=0 and that folds are independent:
    Var(mean of n_folds iid draws of variance sigma^2) = sigma^2/n_folds;
    Var(difference of two such independent means) = 2*sigma^2/n_folds;
    SD = sigma * sqrt(2/n_folds)."""
    return seed_sd * np.sqrt(2.0 / n_folds)


def percentile_rank(value, sample):
    """Percentile (0-100) of `value` within `sample`, via the mean-rank
    (average of strict-less and less-or-equal fractions) convention --
    scipy's default "mean" kind, robust when `value` ties a sample
    member or falls outside the sample's range."""
    return float(scipy_stats.percentileofscore(sample, value, kind="mean"))


if __name__ == "__main__":
    import pandas as pd
    import statsmodels.formula.api as smf

    SCRATCH = "/scratch/project_2019517/tmp_migration_test_hamehd/seed_pilot"
    LOGDIR = "/users/hamehd/projects/sleepfm-mesa/logs"
    import json

    JOB_IDS = {
        ("fromscratch", "EEG1"): "3330256", ("fromscratch", "EEG2"): "3330257",
        ("fromscratch", "EEG3"): "3330258", ("nexttoken", "EEG1"): "3330259",
        ("nexttoken", "EEG2"): "3330260", ("nexttoken", "EEG3"): "3330261",
    }
    MODEL_LABEL = {"fromscratch": "SleepFM From-Scratch", "nexttoken": "SleepFM Next-Token"}
    REFERENCE = {
        ("fromscratch", "EEG1"): 0.6899085410076375, ("fromscratch", "EEG2"): 0.6798274774079068,
        ("fromscratch", "EEG3"): 0.6965673600332758, ("nexttoken", "EEG1"): 0.6960411394299488,
        ("nexttoken", "EEG2"): 0.6963184100797297, ("nexttoken", "EEG3"): 0.698059478557318,
    }
    # results/04's documented 5-fold mean diffs and fold SDs (EEG1=Fz-Cz, EEG2=Cz-Oz, EEG3=C4-M1)
    RESULTS04_GAPS = {
        "fromscratch": {"EEG1-EEG2": -0.0008, "EEG1-EEG3": -0.0051, "EEG2-EEG3": -0.0043},
        "nexttoken": {"EEG1-EEG2": +0.0049, "EEG1-EEG3": +0.0051, "EEG2-EEG3": +0.0002},
    }
    RESULTS04_FOLD_SD = {
        ("fromscratch", "EEG1"): 0.0120, ("fromscratch", "EEG2"): 0.0100, ("fromscratch", "EEG3"): 0.0121,
        ("nexttoken", "EEG1"): 0.0063, ("nexttoken", "EEG2"): 0.0098, ("nexttoken", "EEG3"): 0.0091,
    }

    rows = []
    trajectories = {}  # (pa, ch, seed) -> list[(epoch, val_f1)]
    for (pa, ch), jobid in JOB_IDS.items():
        pilot = json.load(open(f"{SCRATCH}/pilot_{pa}_{ch}.json"))
        log_text = open(f"{LOGDIR}/seedpilot_seedpilot_{jobid}.log").read()
        segments = split_log_into_seed_segments(log_text)
        for r in pilot:
            seed = r["seed"]
            traj = parse_epoch_trajectory(segments[seed])
            trajectories[(pa, ch, seed)] = traj
            stop_epoch, best_epoch, best_val_f1 = parse_stop_and_best_epoch(segments[seed])
            rows.append(dict(model=MODEL_LABEL[pa], pa=pa, channel=ch, group=f"{pa}_{ch}", seed=seed,
                              macro_f1=r["macro_f1"], accuracy=r["accuracy"],
                              stop_epoch=stop_epoch, best_epoch=best_epoch, best_val_f1=best_val_f1))

    df = pd.DataFrame(rows)

    print("=" * 100)
    print("1a. Stop epoch vs. macro F1")
    print("=" * 100)
    rho, p = scipy_stats.spearmanr(df["stop_epoch"], df["macro_f1"])
    print(f"Spearman correlation (n=30): rho={rho:.4f} p={p:.4g}")
    model_fe = smf.ols("macro_f1 ~ stop_epoch + C(group)", data=df).fit()
    slope = model_fe.params["stop_epoch"]
    ci_lo, ci_hi = model_fe.conf_int().loc["stop_epoch"]
    print(f"Fixed-effects OLS (group = model x channel, 6 levels): "
          f"slope(stop_epoch)={slope:.6f}  95% CI=[{ci_lo:.6f}, {ci_hi:.6f}]  "
          f"(per-run macro F1 change per additional stop epoch, within-group)")
    print(f"  p-value on slope: {model_fe.pvalues['stop_epoch']:.4g}  R^2={model_fe.rsquared:.4f}")

    print()
    print("=" * 100)
    print("1b. Peak epoch and last-30-epoch flatness, per run")
    print("=" * 100)
    print(f"{'model':<22} {'ch':<5} {'seed':<5} {'stop_ep':<8} {'peak_ep':<8} {'n_tail':<7} {'val_f1_sd':<10} {'val_f1_range':<12}")
    flat_rows = []
    for row in rows:
        traj = trajectories[(row["pa"], row["channel"], row["seed"])]
        n_used, sd, rng = flatness_last_n_epochs(traj, n=30)
        flat_rows.append(dict(row, flat_n=n_used, flat_sd=sd, flat_range=rng))
        print(f"{row['model']:<22} {row['channel']:<5} {row['seed']:<5} {row['stop_epoch']:<8} "
              f"{row['best_epoch']:<8} {n_used:<7} {sd:<10.5f} {rng:<12.5f}")
    mean_flat_sd = st.mean(r["flat_sd"] for r in flat_rows)
    mean_flat_range = st.mean(r["flat_range"] for r in flat_rows)
    print(f"\nMean last-30-epoch val-F1 SD across all 30 runs: {mean_flat_sd:.5f} "
          f"(range metric mean: {mean_flat_range:.5f}) -- the validation curve is "
          f"{'very flat' if mean_flat_sd < 0.01 else 'moderately flat' if mean_flat_sd < 0.02 else 'not flat'} "
          f"near the stopping point, consistent with patience-10 stopping on noise around a plateau "
          f"rather than catching the loss mid-climb.")

    print()
    print("=" * 100)
    print("1c. Seed SD (95% CI, chi-square n=5) vs. expected 5-fold channel-gap SD vs. results/04")
    print("=" * 100)
    seed_sd_by_group = {}
    for pa in ["fromscratch", "nexttoken"]:
        for ch in ["EEG1", "EEG2", "EEG3"]:
            vals = df[(df["pa"] == pa) & (df["channel"] == ch)]["macro_f1"].tolist()
            sd, lo, hi = chi2_sd_ci(vals)
            exp_gap_sd = expected_fold_gap_sd(sd)
            seed_sd_by_group[(pa, ch)] = sd
            print(f"{MODEL_LABEL[pa]:<22} {ch:<5}: seed_SD={sd:.5f} 95% CI=[{lo:.5f}, {hi:.5f}] | "
                  f"expected 5-fold gap SD (seed_SD*sqrt(2/5))={exp_gap_sd:.5f} | "
                  f"results/04 fold SD={RESULTS04_FOLD_SD[(pa, ch)]:.4f}")
    print()
    for pa in ["fromscratch", "nexttoken"]:
        for pair, gap in RESULTS04_GAPS[pa].items():
            ch_a, ch_b = pair.split("-")
            exp_sd_a = expected_fold_gap_sd(seed_sd_by_group[(pa, ch_a)])
            exp_sd_b = expected_fold_gap_sd(seed_sd_by_group[(pa, ch_b)])
            exp_sd = (exp_sd_a + exp_sd_b) / 2  # average of the two channels' expected-gap SDs
            ratio = abs(gap) / exp_sd if exp_sd > 0 else float("inf")
            print(f"{MODEL_LABEL[pa]:<22} {pair}: results/04 gap={gap:+.4f} | "
                  f"expected gap SD from seed noise={exp_sd:.5f} | "
                  f"|gap|/expected_SD={ratio:.2f} -> "
                  f"{'gap is within 1 SD of pure seed noise' if ratio < 1 else 'gap exceeds 1 seed-noise SD' if ratio < 2 else 'gap exceeds 2 seed-noise SDs'}")

    print()
    print("=" * 100)
    print("1d. Percentile of the unseeded reference run within its 5 seeds")
    print("=" * 100)
    diffs = []
    for pa in ["fromscratch", "nexttoken"]:
        for ch in ["EEG1", "EEG2", "EEG3"]:
            seed_vals = df[(df["pa"] == pa) & (df["channel"] == ch)]["macro_f1"].tolist()
            ref = REFERENCE[(pa, ch)]
            pct = percentile_rank(ref, seed_vals)
            seed_mean = st.mean(seed_vals)
            d = ref - seed_mean
            diffs.append(d)
            print(f"{MODEL_LABEL[pa]:<22} {ch:<5}: reference={ref:.6f} seed_mean={seed_mean:.6f} "
                  f"reference-seed_mean={d:+.6f} percentile_within_seeds={pct:.1f}")
    print(f"\nMean (reference - seed_mean) across all 6 (model,channel) pairs: {st.mean(diffs):+.6f}")
