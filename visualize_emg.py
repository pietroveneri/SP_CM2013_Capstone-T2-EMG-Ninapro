"""Temporary preview of Ninapro DB1: python visualize_emg.py.

Example: python visualize_emg.py --subject 2 --channel 4 --seconds 20
To save only the PNG: python visualize_emg.py --no-show

The Fourier panel uses the same selected channel and time interval.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from scipy.io import loadmat


def fourier_amplitude(signal, fs):
    """One-sided amplitude spectrum of a mean-centered, Hann-windowed signal.

    Normalize by the window sum to preserve bin-centered sinusoid amplitudes.
    DC and the Nyquist bin (when present) are not doubled.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or signal.size < 2 or not np.all(np.isfinite(signal)):
        raise ValueError("Expected a finite, one-dimensional signal with at least two samples")
    if fs <= 0 or not np.isfinite(fs):
        raise ValueError("Sampling rate must be positive and finite")
    n = signal.size
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
    amplitude = np.abs(np.fft.rfft((signal - signal.mean()) * window)) / window.sum()
    amplitude[1:-1 if n % 2 == 0 else None] *= 2
    return np.fft.rfftfreq(n, d=1 / fs), amplitude


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--channel", type=int, choices=range(1, 11), default=1)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--start", type=float, help="Start time in seconds; default: 2 s before the first gesture")
    parser.add_argument("--no-show", action="store_true", help="Save without opening the plot window")
    parser.add_argument("--output", type=Path, default=root / "anteprima_emg.png")
    args = parser.parse_args()
    if args.seconds <= 0 or not np.isfinite(args.seconds):
        parser.error("--seconds must be positive and finite")
    if args.start is not None and (args.start < 0 or not np.isfinite(args.start)):
        parser.error("--start must be non-negative and finite")

    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = root / "NinaPro_Mat" / f"s{args.subject}" / f"S{args.subject}_A1_E1.mat"
    if not path.is_file():
        parser.error(f"File not found: {path}")
    data = loadmat(path)
    emg = np.asarray(data["emg"], dtype=float)
    gesture = data["restimulus"].ravel().astype(int)
    fs = 100.0  # DB1 dataset sampling rate, as in the supplied loader.
    active = np.flatnonzero(gesture > 0)
    start = args.start if args.start is not None else max(0.0, active[0] / fs - 2) if active.size else 0.0
    lo = int(start * fs)
    hi = min(len(emg), lo + max(2, int(args.seconds * fs)))
    if hi - lo < 2:
        parser.error("Time interval outside the recording or too short")
    time = np.arange(lo, hi) / fs
    segment = emg[lo:hi]

    fig, axes = plt.subplots(4, 1, figsize=(13, 11),
                             gridspec_kw={"height_ratios": [2, 2, 1, 2]}, layout="constrained")
    axes[1].sharex(axes[0])
    axes[2].sharex(axes[0])
    fig.suptitle(f"Ninapro DB1 | S{args.subject}, E1 | EMG envelope at 100 Hz\n"
                 "Signal already rectified and filtered; original amplitudes, without normalization",
                 fontsize=13)
    axes[0].plot(time, segment[:, args.channel - 1], color="#0072B2", linewidth=1)
    axes[0].set_title(f"Channel {args.channel}: muscle signal amplitude over time", loc="left")
    axes[0].set_ylabel("Amplitude\n(file units)")
    axes[0].grid(alpha=0.25)

    mesh = axes[1].imshow(segment.T, aspect="auto", origin="lower", interpolation="nearest",
                          extent=(lo / fs, hi / fs, 0.5, 10.5), cmap="viridis", vmin=0)
    axes[1].set_title("All 10 channels: lighter color = higher amplitude", loc="left")
    axes[1].set_ylabel("EMG channel")
    axes[1].set_yticks(range(1, 11))
    fig.colorbar(mesh, ax=axes[:2], label="Amplitude (file units)", shrink=0.8)

    axes[2].step(time, gesture[lo:hi], where="post", color="#D55E00", linewidth=1.5)
    labels = np.unique(np.r_[0, gesture[lo:hi]])
    axes[2].set_yticks(labels, ["0: rest" if g == 0 else f"g{g:02d}" for g in labels])
    axes[2].set_ylim(-0.5, max(1, int(labels.max())) + 0.5)
    axes[2].set_ylabel("True gesture")
    axes[2].set_xlabel("Time from the start of the recording (s)")
    axes[2].set_title("Corrected dataset labels (restimulus), not predictions", loc="left")
    axes[2].grid(alpha=0.25)
    axes[2].set_xlim(lo / fs, hi / fs)

    frequencies, amplitude = fourier_amplitude(segment[:, args.channel - 1], fs)
    axes[3].plot(frequencies, amplitude, color="#009E73", linewidth=1)
    axes[3].set_title(
        f"Channel {args.channel}: Fourier amplitude spectrum | mean removed, Hann window\n"
        f"Envelope frequencies | FFT bin spacing: {fs / len(segment):.3f} Hz", loc="left")
    axes[3].set_xlabel("Frequency (Hz)")
    axes[3].set_ylabel("Amplitude\n(file units)")
    axes[3].set_xlim(0, fs / 2)
    axes[3].set_ylim(bottom=0)
    axes[3].grid(alpha=0.25)

    if args.no_show:
        fig.savefig(args.output, dpi=160)
        print(f"Plot saved: {args.output.resolve()}")
    print(f"File: {path.name} | total duration: {len(emg) / fs:.1f} s")
    print(f"Displayed segment: {lo / fs:.2f}-{hi / fs:.2f} s | channel {args.channel}")
    print("The ML loader excludes rest and windows spanning gesture/repetition changes.")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
