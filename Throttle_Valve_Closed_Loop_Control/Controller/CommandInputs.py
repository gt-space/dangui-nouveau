
import numpy as np


def step(timespan, initial_value, final_value, t_step):
    """
    Generate a step input schedule.

    The output follows:
        - initial_value for t < t_step
        - final_value for t >= t_step

    Parameters
    ----------
    timespan : array-like
        Array of time values at which to evaluate the schedule.
    initial_value : float
        Value before the step occurs.
    final_value : float
        Value after the step occurs.
    t_step : float
        Time at which the step occurs.

    Returns
    -------
    np.ndarray
        Array of scheduled values corresponding to the input timespan.
    """
    values = np.zeros_like(timespan, dtype=float)

    for i, t in enumerate(timespan):
        if t < t_step:
            values[i] = initial_value
        else:
            values[i] = final_value

    return values


def ramp(timespan, initial_value, final_value, t1, t2):
    """
    Generate a piecewise time schedule consisting of an initial hold, linear ramp,
    and final hold.

    The output follows:
        - initial_value for t < t1
        - linearly interpolates from initial_value to final_value for t1 <= t < t2
        - final_value for t >= t2

    Parameters
    ----------
    timespan : array-like
        Array of time values at which to evaluate the schedule.
    initial_value : float
        Value held before the ramp begins.
    final_value : float
        Value reached after the ramp completes.
    t1 : float
        Start time of the ramp.
    t2 : float
        End time of the ramp (must be greater than t1).

    Returns
    -------
    np.ndarray
        Array of scheduled values corresponding to the input timespan.

    Raises
    ------
    ValueError
        If t2 <= t1.
    """
    if t2 <= t1:
        raise ValueError("t2 must be greater than t1.")

    values = np.zeros_like(timespan, dtype=float)

    for i, t in enumerate(timespan):
        if t < t1:
            values[i] = initial_value
        elif t < t2:
            frac = (t - t1) / (t2 - t1)
            values[i] = initial_value + frac * (final_value - initial_value)
        else:
            values[i] = final_value

    return values

