#!/usr/bin/env python
"""
basic detectors
"""
import numpy as np
import numpy.typing as npt


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
