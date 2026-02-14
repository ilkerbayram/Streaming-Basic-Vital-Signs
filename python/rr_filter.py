#!/usr/bin/env python
"""
implements the IBI filter based on the inverse Gaussian process,
described in "Interbeat Interval Filtering", Ilker Bayram, 2024.

ibayram@ieee.org
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class RRFilter:
    """
    the parameterization follows the
    conjugate prior in eqn 2.2
    of his thesis
    """

    a: float
    b: float
    c: float
    d: float
    gam: float

    def __post_init__(self):
        """
        set ml estimates of mu and lambda
        """
        self.ml_estimate()

    def propagate(self):
        """
        power method
        """
        self.a = self.gam * self.a
        self.b = self.gam * self.b
        self.c = self.gam * self.c
        self.d = self.gam * self.d

    def update(self, x):
        """
        update statistics given a new observation
        """
        self.d += 1 / 2
        self.a += x / 2
        self.b += 1
        self.c += 1 / (2 * x)

    def ml_estimate(self):
        """
        update ml estimates of mu and lambda
        """
        self.mu = 2 * self.a / self.b
        temp = self.a / self.mu**2 - self.b / self.mu + self.c
        self.lam = self.d / np.maximum(temp, 1e-2)

    def __call__(self, x):
        """
        update parameters using new observation
        and return ml estimates of mu and lambda
        """
        self.propagate()
        self.update(x)
        # update estimates of mu and lambda
        self.ml_estimate()
        return self.mu, self.lam

    def likelihood(self, x):
        """
        likelihood of x
        """
        out = np.sqrt(self.lam / (2 * np.pi * x**3)) * np.exp(
            -self.lam * (x - self.mu) ** 2 / (2 * x * self.mu**2)
        )
        return out


class RRPDAF(RRFilter):
    def __init__(self, a, b, c, d, gam, prob_error, rr_limits=[0.25, 2]):
        super().__init__(a, b, c, d, gam)
        self.prob_error = prob_error
        self.rr_limits = rr_limits

    def get_previous_params(self):
        self.old_a = self.a
        self.old_b = self.b
        self.old_c = self.c
        self.old_d = self.d

    def interpolate(self, prob_anomalous):
        self.a += prob_anomalous * (self.old_a - self.a)
        self.b += prob_anomalous * (self.old_b - self.b)
        self.c += prob_anomalous * (self.old_c - self.c)
        self.d += prob_anomalous * (self.old_d - self.d)

    def get_prob_anomalous(self, x):
        if (x > self.rr_limits[0]) & (x < self.rr_limits[1]):
            p_anomalous = self.prob_error / (
                self.rr_limits[1] - self.rr_limits[0]
            )
            p_not_anomalous = (1 - self.prob_error) * self.likelihood(x)
            return p_anomalous / (p_anomalous + p_not_anomalous)
        else:
            return 1.0

    def __call__(self, x):
        self.get_previous_params()
        self.propagate()
        prob_anomalous = self.get_prob_anomalous(x)
        self.update(x)
        self.interpolate(prob_anomalous=prob_anomalous)
        self.ml_estimate()
        return self.mu, self.lam, prob_anomalous


@dataclass
class PeakToHR:
    fs: float

    def __post_init__(self):
        self.time_elapsed_from_last_peak = 0
        N = 100
        K = 10
        self.rr_params = {
            "a": 0.5 * K,
            "b": 1 * K,
            "c": 0.5 * K,
            "d": K / 2,
            "gam": (N - 1) / N,
        }
        self.mu = 0
        self.lam = 0
        self.rr_filter = RRFilter(**self.rr_params)

    def __call__(self, peaks):
        for peak in peaks:
            if not peak:
                self.time_elapsed_from_last_peak += 1
            else:
                RR = self.time_elapsed_from_last_peak / self.fs
                self.time_elapsed_from_last_peak = 0
                self.mu, self.lam = self.rr_filter(RR)
        return self.mu, self.lam