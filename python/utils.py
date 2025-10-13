#!/usr/bin/env python
"""
utilities for ppg processing
"""
import numpy as np
import numpy.typing as npt


class Buffer:
    """
    A simple buffer to store data temporarily.
    When the buffer exceeds 200 items, it removes the oldest item.
    When called, it returns the current data as a numpy array and clears the buffer.
    """

    def __init__(self):
        self.data = []

    def append(self, data):
        self.data.append(data)
        if len(self.data) > 200:
            print("Buffer full, popping")
            self.data.pop(0)

    def __call__(self):
        buf = np.array(self.data)
        self.data = []
        return buf


class Spectrogram:
    """
    Compute spectrogram of a 1D signal using STFT.

    Parameters:
    fs: Sampling frequency of the input signal.
    window_size: Size of the analysis window in samples.
    nfft: Number of FFT points.
    hopsize: Hop size between successive windows in samples.
    freq_range: Frequency range to keep in the output spectrogram.
    """

    def __init__(
        self, fs, window_size: int, nfft: int, hopsize: int, freq_range=[0, 4]
    ):
        self.fs = fs
        self.nfft = nfft
        self.hopsize = hopsize
        self.window_size = window_size
        self.window = np.hanning(window_size)
        self.buffer = np.zeros(window_size)
        self.freq_range = freq_range
        freqs = np.arange(nfft) * fs / nfft
        self.freq_inds = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        self.freqs = freqs[self.freq_inds]

    def __call__(self, x: npt.NDArray):
        self.buffer = np.concatenate([self.buffer, x]).reshape(-1)
        out = []
        while len(self.buffer) >= self.window_size:
            x = self.buffer[: self.window_size]
            self.buffer = self.buffer[self.hopsize :]
            X = np.fft.fft(x * self.window, self.nfft)
            out.append(np.abs(X[self.freq_inds]))
        return np.array(out).transpose()
