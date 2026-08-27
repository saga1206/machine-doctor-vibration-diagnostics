"""
Vibration Analysis Module (Step 5)

Kid explanation:
    Step 3 gave us a list of numbers over time -- "how much did it move
    at each moment." That's like a jump rope's up-and-down position
    recorded every instant. This module answers two simple questions
    about that jump rope:

      1. How many times per second does it bounce?  -> FREQUENCY (Hz)
      2. How high does it bounce?                    -> AMPLITUDE

    We use FFT (Fast Fourier Transform) to answer question 1. FFT takes
    a wiggly signal and tells you which "beat" (frequency) it's made of,
    even if the signal looks noisy and complicated to the human eye.

Why frequency AND amplitude both matter for diagnosis:
    Real machine fault patterns are described by BOTH. e.g. "high
    frequency + low amplitude" often means early bearing wear, while
    "low frequency + high amplitude" often means a loose or unbalanced
    part. Frequency alone, or amplitude alone, isn't enough to tell
    these apart -- Step 7 (diagnosis) needs both numbers from this
    module.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class VibrationFeatures:
    dominant_frequency_hz: float
    amplitude: float          # peak amplitude of the dominant vibration
    rms_amplitude: float      # overall "energy" of the whole signal
    spectrum_freqs: np.ndarray = field(repr=False)
    spectrum_magnitudes: np.ndarray = field(repr=False)
    secondary_peaks_hz: list = field(default_factory=list)
    success: bool = True
    error_message: str = ""


def analyze_vibration(signal, fps, freq_min_hz=0.5, freq_max_hz=None) -> VibrationFeatures:
    """
    Main entry point for Step 5.

    `signal` is the filtered per-frame motion signal from Step 3
    (ego_motion_filter's `filtered_signal`). `fps` is the video's frame
    rate, needed to convert "cycles per frame" into real Hz.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)

    if n < 8:
        return VibrationFeatures(
            dominant_frequency_hz=0, amplitude=0, rms_amplitude=0,
            spectrum_freqs=np.array([]), spectrum_magnitudes=np.array([]),
            success=False, error_message="Signal too short for FFT (need at least 8 samples).",
        )

    nyquist = fps / 2.0
    if freq_max_hz is None:
        freq_max_hz = nyquist * 0.95

    # Remove DC offset: don't count overall drift as "vibration"
    signal_centered = signal - np.mean(signal)

    # Hann window: fades the start/end to zero so FFT doesn't get
    # confused by the signal not looping perfectly (reduces "spectral
    # leakage" that would otherwise smear the frequency reading).
    window = np.hanning(n)
    windowed = signal_centered * window

    fft_result = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    magnitudes = np.abs(fft_result)

    # Rescale so magnitude approximates real physical amplitude
    window_correction = 2.0 / np.sum(window)
    magnitudes_calibrated = magnitudes * window_correction

    valid_mask = (freqs >= freq_min_hz) & (freqs <= freq_max_hz)
    if not np.any(valid_mask):
        return VibrationFeatures(
            dominant_frequency_hz=0, amplitude=0,
            rms_amplitude=float(np.sqrt(np.mean(signal_centered ** 2))),
            spectrum_freqs=freqs, spectrum_magnitudes=magnitudes_calibrated,
            success=False,
            error_message=f"No frequency content found between {freq_min_hz}-{freq_max_hz} Hz.",
        )

    valid_freqs = freqs[valid_mask]
    valid_mags = magnitudes_calibrated[valid_mask]

    peak_idx = np.argmax(valid_mags)
    dominant_freq = float(valid_freqs[peak_idx])
    dominant_amplitude = float(valid_mags[peak_idx])

    # Secondary peaks: useful later for spotting harmonics (a real fault
    # often shows up at 1x AND 2x its base frequency)
    secondary_peaks = []
    sorted_indices = np.argsort(valid_mags)[::-1]
    for idx in sorted_indices[:8]:
        f = valid_freqs[idx]
        if abs(f - dominant_freq) > max(1.0, 0.05 * fps):
            secondary_peaks.append(round(float(f), 2))
        if len(secondary_peaks) >= 3:
            break

    rms_amplitude = float(np.sqrt(np.mean(signal_centered ** 2)))

    return VibrationFeatures(
        dominant_frequency_hz=round(dominant_freq, 2),
        amplitude=round(dominant_amplitude, 4),
        rms_amplitude=round(rms_amplitude, 4),
        spectrum_freqs=freqs,
        spectrum_magnitudes=magnitudes_calibrated,
        secondary_peaks_hz=secondary_peaks,
        success=True,
    )