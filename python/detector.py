#!/usr/bin/env python
"""
basic detectors
"""
import numpy as np
import numpy.typing as npt
from collections import deque

class MaxPicker:
    """
    expects a spectrum at each call,
    returns the global maximum in a band of interest
    """

    def __init__(
        self, freqs: npt.NDArray, f_low: float = 0, f_high: float = -1
    ) -> "MaxPicker":
        self.freqs = freqs
        self.f_low = f_low
        self.f_high = f_high

        self.inds = (self.freqs >= self.f_low) & (self.freqs <= self.f_high)
        if not np.any(self.inds):
            self.inds = np.ones_like(self.freqs.shape, dtype=bool)

    def __call__(self, spect):
        vals = spect[self.inds]
        max_ind = np.argmax(np.abs(vals))
        return self.freqs[self.inds][max_ind]

class MaxFilter:
    """
    implements a sliding window maximum filter
    the class keeps track of an integer for indexing, which keeps increasing.
    a safer but less efficient approach is to update the index values in the deque
    """

    def __init__(self, win_len: int):
        self.win_len = win_len
        self.index = 0
        self.index_buf = deque()
        self.val_buf = deque()
        self.delay = self.win_len // 2

    def __call__(self, val: float):
        # if deque is empty, just insert the element and report it

        if not self.index_buf:
            self.index_buf.append(self.index)
            self.val_buf.append(val)
            self.index += 1
            return val

        # oherwise, first pop elements that are out of range
        while self.index_buf and (self.index_buf[0] <= (self.index - self.win_len)):
            self.index_buf.popleft()
            self.val_buf.popleft()
        # if there are any elements left insert value from the right
        while self.index_buf and (val > self.val_buf[-1]):
            self.index_buf.pop()
            self.val_buf.pop()

        # now insert the value
        self.index_buf.append(self.index)
        self.val_buf.append(val)
        self.index += 1
        return self.val_buf[0]
    
class HaarWavelet:
    """
    responsible for ouputting a Haar UDWT
    for a streaming signal, with a certain delay
    """

    def __init__(self, num_levels):
        self.num_levels = num_levels

        self.buffer_size = 2 ** num_levels
        self.buffer = np.zeros(self.buffer_size - 1)
        self.mid_pt = 2 ** (num_levels - 1)

    def __call__(self, new_sample: float):
        # update_buffer
        self.buffer = np.concatenate([self.buffer, [new_sample]])
        # get the coefficients - note we don't need to explicitly obtain filter coefficients thanks to the simplicity of the Haar wavelets
        coefs = [
            (
                np.sum(self.buffer[(self.mid_pt - 2 ** i) : self.mid_pt])
                - np.sum(self.buffer[self.mid_pt : (self.mid_pt + 2 ** i)])
            )
            / 2 ** ((i + 1) / 2)
            for i in range(self.num_levels)
        ]
        self.buffer = self.buffer[1:]
        return coefs
    
class QRSDetector:
  def __init__(self, fs:float, max_level:int=5, thold:float=0.2, winsize:float=0.1):
    self.max_level = max_level
    self.haar_udwt = HaarWavelet(max_level)
    self.haar_delay = self.haar_udwt.mid_pt
    self.fs = fs
    self.max_filter = MaxFilter(int(winsize*fs))
    self.max_filter_delay = self.max_filter.delay
    self.delay = self.max_filter_delay + self.haar_delay
    self.thold = thold
    self.winsize = winsize
    self.coef_buffer = deque()
    for i in range(self.max_filter_delay):
      self.coef_buffer.append(0)

  def __call__(self, new_sample):
    new_coefs = self.haar_udwt(new_sample)
    new_prod = np.prod(np.abs(new_coefs))
    self.coef_buffer.append(new_prod)
    new_max = self.max_filter(new_prod)

    out_coef = self.coef_buffer.popleft()
    return (out_coef == new_max) & (out_coef > self.thold)

class TimeDomainPeakDetector:
    """
    A simple time-domain peak detector using a thresholding method.
    """

    def __init__(self, fs:float=200.0) -> "TimeDomainPeakDetector":
        self.fs = fs
        self.threshold = 1.0 # adaptive threshold, will be updated
        t = np.arange(-0.5, 0.5, 1/fs)
        sigma = 0.1
        self.derivative_window = (1 - (t/sigma)**2) * np.exp(-(t**2)/(2*sigma**2))
        self.buffer_length = len(self.derivative_window) - 1
        self.buffer = np.zeros(self.buffer_length)
        self.delay = self.buffer_length // 2
        self.energy_smoother = np.hanning(int(1.5 * fs))
        self.energy_smoother /= np.sum(self.energy_smoother)
        self.max_guard = int(0.2 * fs)  # 200 ms refractory period
        self.magnitude_buffer = np.zeros(self.max_guard)
        self.greater_than_past = np.zeros(self.max_guard, dtype=bool)

    def get_local_peaks(self, data: npt.NDArray) -> npt.NDArray:
        buffer = np.concatenate([self.magnitude_buffer, data])
        # check if greater than past max_guard samples
        for i in range(len(data)-1, len(buffer)):
            self.greater_than_past[i - len(data)] = np.all(
                data[i - len(data)] > buffer[i - len(data) : i]
            )
        peak_inds = []
        for i in range(len(data) - self.max_guard):
            window = buffer[i : i + 2 * self.max_guard]
            local_max_ind = np.argmax(window)
            if local_max_ind == 0:
                peak_inds.append(i)
    def __call__(self, new_samples: npt.NDArray) -> npt.NDArray:
        """
        Process new samples and detect peaks.

        Parameters:
        new_samples: New time-domain samples to process.

        Returns:
        Detected peak locations in the new samples.
        """
        # Concatenate buffer with new samples
        buffer = np.concatenate([self.buffer, new_samples])
        # Compute derivative using convolution
        derivative = np.convolve(buffer, self.derivative_window, mode='valid')
        magnitude = derivative**2
        # Update buffer for next call
        self.buffer = buffer[-self.buffer_length:]
        self.energy = np.convolve(magnitude, self.energy_smoother, mode='valid')
        median_value = np.median(self.energy)
        max_value = np.max(self.energy)
        self.threshold = 0.3 * max_value + 0.7 * median_value
        
        return magnitude
        # Detect peaks above threshold
        peak_inds = np.where(derivative > self.threshold)[0]
        # Update threshold adaptively (simple moving average)
        if len(derivative) > 0:
            self.threshold = 0.9 * self.threshold + 0.1 * np.mean(np.abs(derivative))
        return peak_inds

    