import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "analysis"))

import compute_profile as cp  # noqa: E402


def test_count_params_bare_state_dict():
    import torch
    sd = {"w": torch.zeros(10, 5), "b": torch.zeros(5)}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        torch.save(sd, path)
        total, size, weights_only, extra = cp.count_params(path)
        assert total == 10 * 5 + 5
        assert size == os.path.getsize(path)
        assert weights_only <= size  # bare state_dict: no extra saved state beyond torch.save's own container overhead
        assert extra == []


def test_count_params_model_state_dict_wrapper():
    import torch
    sd = {"model_state_dict": {"w": torch.zeros(4, 4)}, "optim_dict": {"lr": 0.001}}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        torch.save(sd, path)
        total, size, weights_only, extra = cp.count_params(path)
        assert total == 16
        assert weights_only == 16 * 4  # float32
        assert weights_only < size  # optim_dict inflates the file beyond just the weights
        assert extra == ["optim_dict"]


def test_count_params_state_dict_wrapper():
    import torch
    sd = {"state_dict": {"w": torch.zeros(3, 3)}, "epoch": 5}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        torch.save(sd, path)
        total, size, weights_only, extra = cp.count_params(path)
        assert total == 9
        assert weights_only == 9 * 4
        assert extra == ["epoch"]


def test_run_input_size_matches_known_sampling_rates():
    rows = cp.run_input_size()
    by_model = {r["model"]: r for r in rows}
    assert by_model["SleepFM From-Scratch"]["sampling_rate_hz"] == 128
    assert by_model["SleepFM Next-Token"]["sampling_rate_hz"] == 128
    assert by_model["BIOT"]["sampling_rate_hz"] == 200
    # 128 Hz * 8h * 3600s = 3,686,400 samples/channel/night
    assert by_model["SleepFM From-Scratch"]["samples_per_channel_per_night"] == 128 * 8 * 3600
    # float32 -> 4 bytes/sample
    expected_bytes = 128 * 8 * 3600 * 4
    assert by_model["SleepFM From-Scratch"]["bytes_per_channel_per_night"] == expected_bytes


def test_run_input_size_biot_uses_more_bytes_than_sleepfm():
    rows = cp.run_input_size()
    by_model = {r["model"]: r for r in rows}
    assert by_model["BIOT"]["bytes_per_channel_per_night"] > by_model["SleepFM From-Scratch"]["bytes_per_channel_per_night"]
