"""Shared acoustic accuracy measurements for conversion and research."""

import numpy as np

SCREEN_LIMITS = {"max_abs_error": 0.05, "rmse": 0.005, "minimum_snr_db": 25.0,
                "max_peak_ratio": 1.05, "peak_absolute_allowance": 1e-4,
                "max_spectral_convergence": 0.05, "max_active_log_spectral_rms_db": 1.0,
                "stft_window": 1024, "stft_hop": 256, "active_magnitude_floor_db": -60.0}
ORIGINAL_TOLERANCE = {"atol": 1e-4, "rtol": 1e-5}


def compare(actual, expected):
    if actual.shape != expected.shape:
        return {"passed": False, "actual_shape": list(actual.shape), "expected_shape": list(expected.shape)}
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return {"passed": False, "finite": False}
    error = actual.astype(np.float64) - expected.astype(np.float64)
    outside = np.abs(error) > ORIGINAL_TOLERANCE["atol"] + ORIGINAL_TOLERANCE["rtol"] * np.abs(expected)
    return {"passed": not bool(np.any(outside)), "max_abs_error": float(np.abs(error).max()),
            "rmse": float(np.sqrt(np.mean(error * error))), "outside_tolerance": int(outside.sum()),
            "elements": int(actual.size), **ORIGINAL_TOLERANCE}


def waveform_metrics(actual, expected):
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return {"passed": False, "shape_equal": actual.shape == expected.shape,
                "finite": bool(np.isfinite(actual).all())}
    actual, expected = actual.astype(np.float64).reshape(-1), expected.astype(np.float64).reshape(-1)
    error = actual - expected
    mse, energy = float(np.mean(error * error)), float(np.mean(expected * expected))
    snr = float(10 * np.log10(max(energy, 1e-30) / max(mse, 1e-30)))
    peak, reference_peak = float(np.max(np.abs(actual))), float(np.max(np.abs(expected)))
    def magnitude(audio):
        width, hop = SCREEN_LIMITS["stft_window"], SCREEN_LIMITS["stft_hop"]
        if len(audio) < width:
            audio = np.pad(audio, (0, width - len(audio)))
        frames = np.lib.stride_tricks.sliding_window_view(audio, width)[::hop]
        return np.abs(np.fft.rfft(frames * np.hanning(width), axis=-1))
    spectrum, reference_spectrum = magnitude(actual), magnitude(expected)
    denominator = float(np.linalg.norm(reference_spectrum))
    convergence = float(np.linalg.norm(spectrum - reference_spectrum)) / max(denominator, 1e-30)
    active = reference_spectrum > max(float(reference_spectrum.max()) *
        10 ** (SCREEN_LIMITS["active_magnitude_floor_db"] / 20), 1e-12)
    log_delta = 20 * np.log10(np.maximum(spectrum[active], 1e-12) /
                             np.maximum(reference_spectrum[active], 1e-12))
    log_rms = float(np.sqrt(np.mean(log_delta * log_delta))) if active.any() else 0.0
    metrics = {"shape_equal": True, "finite": True, "samples": int(actual.size),
               "max_abs_error": float(np.abs(error).max()), "rmse": float(np.sqrt(mse)),
               "snr_db": snr, "peak": peak, "reference_peak": reference_peak,
               "rms": float(np.sqrt(np.mean(actual * actual))), "reference_rms": float(np.sqrt(energy)),
               "absolute_peak_delta": peak - reference_peak,
               "spectral_convergence": convergence, "active_log_spectral_rms_db": log_rms,
               "active_spectral_bins": int(active.sum()),
               "samples_outside_unit_range": int(np.count_nonzero(np.abs(actual) > 1)),
               "reference_samples_outside_unit_range": int(np.count_nonzero(np.abs(expected) > 1))}
    checks = {"max_abs_error": metrics["max_abs_error"] <= SCREEN_LIMITS["max_abs_error"],
              "rmse": metrics["rmse"] <= SCREEN_LIMITS["rmse"],
              "snr": snr >= SCREEN_LIMITS["minimum_snr_db"] or mse == 0,
              "amplitude": peak <= reference_peak * SCREEN_LIMITS["max_peak_ratio"] + SCREEN_LIMITS["peak_absolute_allowance"],
              "spectral_convergence": convergence <= SCREEN_LIMITS["max_spectral_convergence"],
              "active_log_spectral_rms": log_rms <= SCREEN_LIMITS["max_active_log_spectral_rms_db"]}
    metrics.update(checks=checks, passed=all(checks.values()))
    return metrics
