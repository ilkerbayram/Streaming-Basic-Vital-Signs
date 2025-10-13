#!/usr/bin/env python
"""
trackers for robust frequency tracking
"""
import numpy as np


class Gaussian:
    def __init__(self, mean: float = 0.0, variance: float = 1.0) -> "Gaussian":
        self.update_statistics(new_mean=mean, new_variance=variance)

    def update_const(self) -> None:
        self.const = 1 / (2 * np.pi * np.sqrt(self.variance))

    def __call__(self, val: float) -> float:
        return np.exp(-((val - self.mean) ** 2) / (2 * self.variance)) * self.const

    def update_statistics(self, new_mean: float, new_variance: float) -> None:
        self.mean = new_mean
        self.variance = new_variance
        self.update_const()


class RobustTracker:
    def __init__(
        self,
        state_update_noise_variance: float = 1e-3,
        observation_noise_variance: float = 1e-3,
        probability_of_detection: float = 0.5,
        min_observation: float = 0.5,
        max_observation: float = 4,
    ) -> "RobustTracker":
        self.state_update_noise_variance = state_update_noise_variance
        self.observation_noise_variance = observation_noise_variance
        self.probability_of_detection = probability_of_detection
        self.min_observation = min_observation
        self.max_observation = max_observation
        self.uniform_pdf = 1 / (self.max_observation - self.min_observation)
        self.state = Gaussian(mean=1.0, variance=1e-2)

    def __call__(self, new_obs: float) -> float:
        z_hat = self.state.mean
        P = self.state.variance + self.state_update_noise_variance
        S = P + self.observation_noise_variance
        beta = np.zeros(2)
        beta[0] = self.uniform_pdf * (1 - self.probability_of_detection)
        measurement_distribution = Gaussian(z_hat, S)
        beta[1] = self.probability_of_detection * measurement_distribution(new_obs)
        beta = beta / np.sum(beta)
        innovation = new_obs - z_hat
        kalman_gain = P / S
        new_mean = self.state.mean + beta[1] * kalman_gain * innovation
        cov1 = max(P - P**2 / S, 0)
        excess_cov = np.prod(beta) * (kalman_gain * innovation) ** 2
        new_variance = beta[0] * P + beta[1] * cov1 + excess_cov
        self.state.update_statistics(new_mean, new_variance)
        return self.state.mean
