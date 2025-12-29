
import numpy as np
import pytest
from backend.pipeline.detectors import create_harmonic_mask

def legacy_create_harmonic_mask(
    f0_hz: np.ndarray,
    sr: int,
    n_fft: int,
    mask_width: float = 0.03,
    n_harmonics: int = 8,
    min_band_hz: float = 6.0,
    frequency_aware_width: bool = False,
) -> np.ndarray:
    """
    Legacy implementation of harmonic mask creation.
    """
    f0_hz = np.asarray(f0_hz, dtype=np.float32).reshape(-1)
    n_frames = int(f0_hz.shape[0])
    n_bins = n_fft // 2 + 1

    mask_T = np.ones((n_frames, n_bins), dtype=np.float32)

    bin_hz = float(sr) / float(n_fft)

    for t in range(n_frames):
        f0 = float(f0_hz[t])
        if f0 <= 0.0 or not np.isfinite(f0):
            continue

        for h in range(1, n_harmonics + 1):
            fh = f0 * h
            if fh >= float(sr) / 2.0:
                break

            if frequency_aware_width:
                 width_factor = 1.0
                 if fh < 200.0:
                     width_factor = 1.0 + (200.0 - fh) / 100.0
                 bw = max(float(min_band_hz), abs(mask_width * width_factor) * fh)
            else:
                bw = max(float(min_band_hz), abs(mask_width) * fh)

            lo = fh - bw
            hi = fh + bw

            idx_lo = int(np.ceil(lo / bin_hz))
            idx_hi = int(np.floor(hi / bin_hz))

            idx_lo = max(0, idx_lo)
            idx_hi = min(n_bins - 1, idx_hi)

            if idx_lo <= idx_hi:
                mask_T[t, idx_lo : idx_hi + 1] = 0.0

    return mask_T.T

def test_harmonic_mask_random_equivalence():
    np.random.seed(42)
    n_frames = 100
    sr = 22050
    n_fft = 2048

    # Generate random f0 between 50 and 1000 Hz, with some zeros/negatives/nans
    f0 = np.random.uniform(50, 1000, n_frames).astype(np.float32)
    f0[0] = 0.0
    f0[1] = -50.0
    f0[2] = np.nan
    f0[3] = np.inf

    mask_width = 0.03
    n_harmonics = 8

    # Test standard mode
    mask_legacy = legacy_create_harmonic_mask(f0, sr, n_fft, mask_width, n_harmonics)
    mask_new = create_harmonic_mask(f0, sr, n_fft, mask_width, n_harmonics)

    np.testing.assert_allclose(mask_new, mask_legacy, atol=1e-7, err_msg="Standard mode mismatch")

    # Test frequency aware
    mask_legacy_fa = legacy_create_harmonic_mask(f0, sr, n_fft, mask_width, n_harmonics, frequency_aware_width=True)
    mask_new_fa = create_harmonic_mask(f0, sr, n_fft, mask_width, n_harmonics, frequency_aware_width=True)

    np.testing.assert_allclose(mask_new_fa, mask_legacy_fa, atol=1e-7, err_msg="Freq aware mismatch")

def test_harmonic_mask_edge_cases():
    sr = 1000
    n_fft = 100
    f0 = np.array([500.0, 501.0], dtype=np.float32) # Near/Above Nyquist (500)

    mask_legacy = legacy_create_harmonic_mask(f0, sr, n_fft)
    mask_new = create_harmonic_mask(f0, sr, n_fft)

    np.testing.assert_allclose(mask_new, mask_legacy, atol=1e-7)

def test_boundary_conditions():
    # Test exact bin boundaries if possible
    sr = 100
    n_fft = 100
    bin_hz = 1.0 # 100/100

    # f0 such that harmonics land exactly on bin edges?
    # bin centers are 0, 1, 2...
    # if harmonic band is [1.5, 2.5], bins 2 should be masked?
    # lo=1.5 -> ceil(1.5) = 2. hi=2.5 -> floor(2.5) = 2.

    f0 = np.array([2.0], dtype=np.float32)
    # 1st harmonic = 2.0. mask_width=0.0 -> bw = min_band_hz.
    min_band = 0.5
    mask_legacy = legacy_create_harmonic_mask(f0, sr, n_fft, min_band_hz=min_band, mask_width=0.0)
    mask_new = create_harmonic_mask(f0, sr, n_fft, min_band_hz=min_band, mask_width=0.0)

    np.testing.assert_allclose(mask_new, mask_legacy, atol=1e-7)

def test_batching_equivalence():
    np.random.seed(123)
    n_frames = 5000 # Enough to trigger batching if batch_size=2000
    sr = 22050
    n_fft = 512
    f0 = np.random.uniform(100, 400, n_frames).astype(np.float32)

    mask_ref = create_harmonic_mask(f0, sr, n_fft, batch_size=50000) # One giant batch
    mask_batched = create_harmonic_mask(f0, sr, n_fft, batch_size=2000) # Small batches

    np.testing.assert_array_equal(mask_batched, mask_ref)
