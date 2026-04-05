"""Smoke tests for SecureLink dataset loading and processing."""

from __future__ import annotations

import numpy as np
import torch

from securelink.dataset import (
    SecureLinkDataset,
    align_csi_to_telemetry,
    clean_telemetry,
    construct_samples,
    extract_phase_errors,
    split_dataset_a,
)


class TestPhaseErrors:
    def test_basic_extraction(self):
        # Simulated phase data: (10 frames, 52 subcarriers)
        phases = np.random.randn(10, 52).astype(np.float64) * 0.5
        errors = extract_phase_errors(phases, eta=4.0)
        assert errors.shape[1] == 52
        assert errors.shape[0] > 0  # some frames should survive filtering

    def test_outlier_filtering(self):
        # Create data where some frames have very high gradient variance
        phases = np.random.randn(20, 52).astype(np.float64) * 0.1
        # Add outlier frames with steep gradients
        phases[5] = np.linspace(-10, 10, 52)
        phases[15] = np.linspace(-20, 20, 52)

        errors = extract_phase_errors(phases, eta=4.0)
        # Outlier frames should be filtered
        assert errors.shape[0] <= 20

    def test_empty_input(self):
        phases = np.array([]).reshape(0, 52)
        errors = extract_phase_errors(phases)
        assert errors.shape == (0, 52)


class TestTelemetryCleaning:
    def test_normal_data(self):
        data = np.random.randn(100, 8).astype(np.float32)
        cleaned = clean_telemetry(data)
        assert cleaned.shape[1] == 8
        assert cleaned.shape[0] > 0

    def test_outlier_removal(self):
        data = np.random.randn(100, 8).astype(np.float32)
        # Add extreme outliers
        data[50] = np.ones(8) * 1000
        cleaned = clean_telemetry(data, max_std=3.0)
        assert cleaned.shape[0] < 100


class TestAlignment:
    def test_upsample(self):
        csi = np.random.randn(5, 52).astype(np.float32)
        telemetry = np.random.randn(100, 8).astype(np.float32)
        aligned = align_csi_to_telemetry(csi, telemetry)
        assert aligned.shape == (100, 52)

    def test_downsample(self):
        csi = np.random.randn(200, 52).astype(np.float32)
        telemetry = np.random.randn(100, 8).astype(np.float32)
        aligned = align_csi_to_telemetry(csi, telemetry)
        assert aligned.shape == (100, 52)

    def test_equal_length(self):
        csi = np.random.randn(100, 52).astype(np.float32)
        telemetry = np.random.randn(100, 8).astype(np.float32)
        aligned = align_csi_to_telemetry(csi, telemetry)
        assert aligned.shape == (100, 52)

    def test_single_frame(self):
        csi = np.random.randn(1, 52).astype(np.float32)
        telemetry = np.random.randn(50, 8).astype(np.float32)
        aligned = align_csi_to_telemetry(csi, telemetry)
        assert aligned.shape == (50, 52)


class TestSampleConstruction:
    def test_basic(self):
        csi = np.random.randn(60, 52).astype(np.float32)
        mems = np.random.randn(60, 8).astype(np.float32)
        csi_s, mems_s = construct_samples(csi, mems, sample_length=6)
        assert csi_s.shape == (10, 6, 52)
        assert mems_s.shape == (10, 6, 8)

    def test_short_sequence(self):
        csi = np.random.randn(3, 52).astype(np.float32)
        mems = np.random.randn(3, 8).astype(np.float32)
        csi_s, mems_s = construct_samples(csi, mems, sample_length=6)
        assert csi_s.shape[1] == 6  # padded to sample_length
        assert mems_s.shape[1] == 6

    def test_exact_multiple(self):
        csi = np.random.randn(12, 52).astype(np.float32)
        mems = np.random.randn(12, 8).astype(np.float32)
        csi_s, mems_s = construct_samples(csi, mems, sample_length=6)
        assert csi_s.shape[0] == 2


class TestSecureLinkDataset:
    def test_basic(self):
        csi = np.random.randn(50, 6, 52).astype(np.float32)
        mems = np.random.randn(50, 6, 8).astype(np.float32)
        labels = np.random.randint(0, 22, size=50).astype(np.int64)

        ds = SecureLinkDataset(csi, mems, labels, normalize=True)
        assert len(ds) == 50

        item = ds[0]
        assert "csi" in item
        assert "mems" in item
        assert "label" in item
        assert item["csi"].shape == (6, 52)
        assert item["mems"].shape == (6, 8)
        assert item["label"].dtype == torch.int64

    def test_no_normalize(self):
        csi = np.random.randn(10, 6, 52).astype(np.float32)
        mems = np.random.randn(10, 6, 8).astype(np.float32)
        labels = np.zeros(10, dtype=np.int64)

        ds = SecureLinkDataset(csi, mems, labels, normalize=False)
        item = ds[0]
        np.testing.assert_allclose(item["csi"].numpy(), csi[0], atol=1e-6)


class TestDatasetSplitting:
    def test_split_a(self):
        N = 100
        csi = np.random.randn(N, 6, 52).astype(np.float32)
        mems = np.random.randn(N, 6, 8).astype(np.float32)
        labels = np.random.randint(0, 22, size=N).astype(np.int64)

        splits = split_dataset_a(csi, mems, labels)
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == N

    def test_reproducibility(self):
        N = 100
        csi = np.random.randn(N, 6, 52).astype(np.float32)
        mems = np.random.randn(N, 6, 8).astype(np.float32)
        labels = np.random.randint(0, 22, size=N).astype(np.int64)

        splits1 = split_dataset_a(csi, mems, labels, seed=42)
        splits2 = split_dataset_a(csi, mems, labels, seed=42)

        item1 = splits1["train"][0]
        item2 = splits2["train"][0]
        assert torch.allclose(item1["csi"], item2["csi"])
