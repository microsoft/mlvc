# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

import numpy as np
import scipy.interpolate


def calculate_bd_rate(
    r1, metric1, r2, metric2, *, piecewise=0, lower_bound: Optional[float] = None, upper_bound: Optional[float] = None
):
    lR1 = np.log(r1)
    lR2 = np.log(r2)

    # integration interval
    min_int = max(min(metric1), min(metric2))
    max_int = min(max(metric1), max(metric2))

    if lower_bound is not None:
        min_int = max(lower_bound, min_int)
    if upper_bound is not None:
        max_int = min(upper_bound, max_int)

    # find integral
    if piecewise == 0:
        # rate method
        p1 = np.polyfit(metric1, lR1, 3)
        p2 = np.polyfit(metric2, lR2, 3)

        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(metric1), np.sort(lR1), samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(metric2), np.sort(lR2), samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapezoid(v1, dx=float(interval))
        int2 = np.trapezoid(v2, dx=float(interval))

    # find avg diff
    avg_exp_diff = (int2 - int1) / (max_int - min_int)
    avg_diff = (np.exp(avg_exp_diff) - 1) * 100
    return avg_diff
