"""
This is a Python script for signal declipping based on Pavel Zaviska's 2020 GitHub code.
The script includes functions for hard clipping a signal, calculating the Signal-to-Distortion Ratio (SDR),
and clipping a signal to a desired SDR value.

Author: Michal Svento
Date: 24.4.2024
"""

import numpy as np
from scipy.optimize import minimize_scalar


def hard_clip(signal, t_max, t_min=None):
    if t_min is None:
        t_min = -t_max

    signal_min = np.min(signal)
    signal_max = np.max(signal)
    if signal_min >= t_min and signal_max <= t_max:
        print('Clipping range too large. No clipping will occur!')
    if t_min >= t_max:
        print('Lower clipping level must be smaller than the upper one!')

    damaged = signal.copy()
    masks = np.ones_like(damaged)
    masks[signal > t_max] = 0
    masks[signal < t_min] = 0

    damaged[signal > t_max] = t_max
    damaged[signal < t_min] = t_min

    return damaged, masks


def sdr(original, degraded):
    return 20 * np.log10(np.linalg.norm(original) / np.linalg.norm(original - degraded))


def clip_sdr(signal, desired_sdr):
    diff_function = lambda t: (sdr(signal, hard_clip(signal, t)[0]) - desired_sdr) ** 2
    clipping_threshold = minimize_scalar(diff_function,
                                         bounds=(np.finfo(np.float32).eps, 0.99 * np.max(np.abs(signal))))

    clipped, masks = hard_clip(signal, clipping_threshold.x)
    percentage = (1 - np.sum(masks) / len(signal)) * 100
    return clipped, masks, clipping_threshold.x, percentage
