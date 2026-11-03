"""Shared library for the Moody/PhysioNet 2026 label-alignment diagnostic
(lag scans, band power, code identification, CAISR cross-check). Read-only
analysis code -- never writes to MESA/SHHS/MrOS data, never shifts labels.

Log band power via Welch, z-scored per record (over scored epochs only),
AUROC-based alignment statistic. This replaced an earlier, broken statistic
(raw non-log Pearson correlation on skewed power values), which failed even
on MESA -- see results/detailed notes for the full diagnostic writeup.
"""
import numpy as np
from scipy.signal import welch
from sklearn.metrics import roc_auc_score

SR = 128
EPOCH_SEC = 30
EPOCH_SAMPLES = SR * EPOCH_SEC

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}


def band_powers_per_epoch(signal, bands=BANDS):
    """Returns dict[band_name] -> np.array of log power per 30s epoch."""
    n_epochs = len(signal) // EPOCH_SAMPLES
    out = {name: np.full(n_epochs, np.nan) for name in bands}
    for i in range(n_epochs):
        seg = signal[i * EPOCH_SAMPLES:(i + 1) * EPOCH_SAMPLES]
        freqs, psd = welch(seg, fs=SR, nperseg=min(len(seg), SR * 4))
        for name, (lo, hi) in bands.items():
            band_mask = (freqs >= lo) & (freqs <= hi)
            power = np.trapezoid(psd[band_mask], freqs[band_mask])
            out[name][i] = np.log(power + 1e-12)
    return out


def zscore_valid(x, valid):
    """Z-score x using only valid (scored) entries' mean/std; applies
    the same transform to all entries (valid or not)."""
    mu = np.mean(x[valid])
    sd = np.std(x[valid])
    if sd == 0 or not np.isfinite(sd):
        return np.zeros_like(x)
    return (x - mu) / sd


def auroc_lag_scan(indicator, score, valid, lags):
    """indicator: binary target (e.g. is-N3) aligned to label indices.
    score: continuous predictor (e.g. z-scored log delta power) aligned
    to SIGNAL indices (same length as indicator here, pre-shift).
    valid: boolean, True where the LABEL at that index is scored
    (StageNumber != -1) -- aligned to indicator's own indexing.
    Returns np.array of AUROC per lag (NaN where undefined)."""
    n = len(indicator)
    aurocs = []
    for lag in lags:
        if lag >= 0:
            ind = indicator[lag:]
            sc = score[:n - lag] if lag > 0 else score
            val = valid[lag:]
        else:
            ind = indicator[:n + lag]
            sc = score[-lag:]
            val = valid[:n + lag]
        ind_v = ind[val]
        sc_v = sc[val]
        if len(ind_v) < 10 or len(np.unique(ind_v)) < 2:
            aurocs.append(np.nan)
            continue
        try:
            aurocs.append(roc_auc_score(ind_v, sc_v))
        except ValueError:
            aurocs.append(np.nan)
    return np.array(aurocs)
