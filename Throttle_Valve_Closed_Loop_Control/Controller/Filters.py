import numpy as np


def rate_limit(prev_value, target_value, dt, max_rate):
    """
    Apply a symmetric rate limiter to a signal.

    This function constrains how quickly a value can change between time steps,
    enforcing a maximum rate of increase or decrease. It is commonly used to
    model actuator slew rate limits (e.g., valve motion) or to prevent abrupt
    command changes in control systems.

    The update follows:
        delta_max = max_rate * dt
        output = prev_value + clip(target_value - prev_value, -delta_max, delta_max)

    Parameters
    ----------
    prev_value : float
        Value from the previous time step.
    target_value : float
        Desired (unconstrained) value at the current time step.
    dt : float
        Time step size [s].
    max_rate : float
        Maximum allowed rate of change [units/s].

    Returns
    -------
    float
        Rate-limited value for the current time step.

    Notes
    -----
    - The limiter is symmetric: the same rate bound is applied to increases
      and decreases.
    - If |target_value - prev_value| <= max_rate * dt, the target is reached
      exactly in one step.
    - Otherwise, the output moves toward the target at the maximum allowed rate.
    """
    delta_max = max_rate * dt
    delta = np.clip(target_value - prev_value, -delta_max, delta_max)
    return prev_value + delta


def apply_error_deadband(target: float, measurement: float, tolerance: float) -> float:
    """
    Return an effective target that disables controller action when the error
    magnitude is below a specified tolerance.
    """
    error = target - measurement
    if abs(error) < tolerance:
        return measurement
    return target


def saturation(x, x_min, x_max):
    """
    Clamp a scalar value between lower and upper bounds.

    Parameters
    ----------
    x : float
        Value to be clamped.
    x_min : float
        Lower allowable bound.
    x_max : float
        Upper allowable bound.

    Returns
    -------
    float
        The clamped value:
        - x_min if x < x_min
        - x_max if x > x_max
        - x otherwise

    Raises
    ------
    ValueError
        If x_min > x_max.

    Notes
    -----
    This is a standard saturation function used heavily in control systems.
    It is useful for representing physical limits such as:

    - valve minimum / maximum CdA
    - motor minimum / maximum angle
    - command voltage limits
    - bounded control effort

    Examples
    --------
    >>> saturation(5.0, 0.0, 10.0)
    5.0
    >>> saturation(-2.0, 0.0, 10.0)
    0.0
    >>> saturation(12.0, 0.0, 10.0)
    10.0
    """
    if x_min > x_max:
        raise ValueError(f"x_min must be <= x_max. Got x_min={x_min}, x_max={x_max}.")

    return max(x_min, min(x, x_max))




def low_pass_filter(prev_filtered, measurement, dt, tau):
    """
    First-order low-pass filter (exponential smoothing).

    Parameters
    ----------
    prev_filtered : float
        Previous filtered value (x_f[k-1]).
    measurement : float
        Current raw measurement (x[k]).
    dt : float
        Timestep (seconds).
    tau : float
        Filter time constant (seconds). Larger tau = more smoothing.

    Returns
    -------
    float
        Updated filtered value (x_f[k]).
    """
    alpha = dt / (tau + dt)
    return prev_filtered + alpha * (measurement - prev_filtered)
