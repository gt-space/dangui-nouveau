from .Actuators import saturation
from .Filters import low_pass_filter, rate_limit


class PID:
    """
    Discrete PID controller with output saturation, optional output slew-rate
    limiting, simple anti-windup, and filtered derivative-on-measurement.

    This controller is intended for sampled-time control of a scalar signal:

        error = target - measurement

    The nominal control law is:

        u_unsat = u_bias + Kp * error + Ki * integral - Kd * d(measurement)/dt

    where the derivative is taken on the measurement instead of the error.
    This avoids derivative kick when the target changes suddenly. The raw
    measurement derivative is low-pass filtered before use in the control law
    to reduce sensitivity to numerical noise, sensor noise, and small plant
    oscillations.

    After the nominal control output is formed, the controller applies:
    1. output saturation between u_min and u_max
    2. optional output slew-rate limiting, if du_dt_limit is provided

    The slew-rate limiter constrains how quickly the final controller output can
    change from one update to the next:

        |u[k] - u[k-1]| / dt <= du_dt_limit

    This is useful when the commanded variable should move more gradually, such
    as a throttle-like correction signal.

    Features
    --------
    - Proportional, integral, and derivative control
    - Derivative on measurement
    - First-order low-pass filtering of the derivative estimate
    - Output saturation between u_min and u_max
    - Optional output slew-rate limiting via du_dt_limit
    - Simple anti-windup by freezing the integral when saturated and the error
      would push the controller farther into saturation
    - Optional constant bias term u_bias

    Parameters
    ----------
    Kp : float
        Proportional gain.
    Ki : float
        Integral gain.
    Kd : float
        Derivative gain. This multiplies the filtered derivative of the
        measurement.
    u_min : float
        Minimum allowable controller output.
    u_max : float
        Maximum allowable controller output.
    u_bias : float, optional
        Constant bias added to the controller output. Default is 0.0.
    tau_d : float, optional
        Time constant for the first-order low-pass filter applied to the raw
        measurement derivative. Larger values give more smoothing but reduce
        derivative responsiveness. Must be nonnegative. Default is 0.05.
    du_dt_limit : float or None, optional
        Maximum allowed rate of change of the final controller output, in
        output-units per second. If None, no slew-rate limiting is applied.
        Must be positive if provided. Default is None.

    Attributes
    ----------
    Kp, Ki, Kd : float
        Controller gains.
    u_min, u_max : float
        Output saturation limits.
    u_bias : float
        Constant bias term.
    tau_d : float
        Derivative filter time constant.
    du_dt_limit : float or None
        Output slew-rate limit in output-units per second.
    error : float or None
        Most recent control error, target - measurement.
    integral : float
        Current integral state.
    derivative : float
        Most recent filtered derivative estimate of d(measurement)/dt.
    prev_measurement : float or None
        Previous measurement sample, used to compute the raw derivative.
    prev_u : float or None
        Previous final controller output, used for slew-rate limiting.
    u_unsat : float or None
        Most recent controller output before saturation.
    u_sat : float or None
        Most recent controller output after saturation but before slew-rate limiting.
    u : float or None
        Most recent final controller output after all limiting.

    Notes
    -----
    The derivative term is applied as:

        -Kd * derivative

    because increasing measurement implies the process variable is already
    moving upward, so the derivative term acts to oppose that motion.

    The anti-windup strategy used here is intentionally simple. It prevents the
    integral from growing when the controller output is saturating and the
    current error would drive it farther into the same saturation limit.

    This class assumes the helper functions `low_pass_filter(...)` and
    `saturation(...)` already exist in the surrounding codebase.
    """

    def __init__(
        self,
        Kp,
        Ki,
        Kd,
        u_min,
        u_max,
        u_bias=0.0,
        tau_d=0.05,
        du_dt_limit=None,
    ):
        Kp = float(Kp)
        Ki = float(Ki)
        Kd = float(Kd)
        u_min = float(u_min)
        u_max = float(u_max)
        u_bias = float(u_bias)
        tau_d = float(tau_d)

        if u_min > u_max:
            raise ValueError(
                f"u_min must be <= u_max. Got u_min={u_min}, u_max={u_max}."
            )

        if tau_d < 0.0:
            raise ValueError(f"tau_d must be >= 0. Got tau_d={tau_d}.")

        if du_dt_limit is not None:
            du_dt_limit = float(du_dt_limit)
            if du_dt_limit <= 0.0:
                raise ValueError(
                    f"du_dt_limit must be > 0 when provided. Got {du_dt_limit}."
                )

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.u_min = u_min
        self.u_max = u_max
        self.u_bias = u_bias
        self.tau_d = tau_d
        self.du_dt_limit = du_dt_limit

        self.error = None
        self.integral = 0.0
        self.derivative = 0.0
        self.prev_measurement = None
        self.prev_u = None

        self.u_unsat = None
        self.u_sat = None
        self.u = None

    def reset(self, measurement=None, output=None):
        """
        Reset the controller internal state.

        Parameters
        ----------
        measurement : float or None, optional
            Current measured process value. If provided, it is stored as the
            previous measurement so that the next update does not produce a
            startup derivative spike.
        output : float or None, optional
            Initial previous output for the slew-rate limiter. If provided,
            the next update will limit changes relative to this value. If
            omitted, the first update bypasses slew-rate limiting.
        """
        self.error = None
        self.integral = 0.0
        self.derivative = 0.0
        self.prev_measurement = None if measurement is None else float(measurement)
        self.prev_u = None if output is None else float(output)

        self.u_unsat = None
        self.u_sat = None
        self.u = None

    def update(self, target, measurement, dt):
        """
        Advance the PID controller by one timestep.

        Parameters
        ----------
        target : float
            Desired setpoint.
        measurement : float
            Current measured process value.
        dt : float
            Controller timestep in seconds. Must be positive.

        Returns
        -------
        u : float
            Final controller output after saturation and optional slew-rate limiting.
        error : float
            Current control error, target - measurement.
        integral : float
            Current integral state after anti-windup logic.
        derivative : float
            Current filtered derivative estimate of d(measurement)/dt.

        Notes
        -----
        The raw measurement derivative is computed as:

            d_meas_dt = (measurement - prev_measurement) / dt

        and then filtered with a first-order low-pass filter:

            derivative = low_pass_filter(previous_derivative, d_meas_dt, dt, tau_d)

        The controller then:
        1. computes the nominal unsaturated output
        2. saturates it to [u_min, u_max]
        3. optionally slew-rate limits the result using du_dt_limit
        4. updates the integral only if the controller is not pushing farther
           into saturation
        """
        dt = float(dt)
        target = float(target)
        measurement = float(measurement)

        if dt <= 0.0:
            raise ValueError(f"dt must be positive. Got dt={dt}.")

        error = target - measurement
        self.error = error

        # Derivative on measurement to avoid derivative kick from setpoint steps.
        if self.prev_measurement is None:
            d_meas_dt = 0.0
        else:
            d_meas_dt = (measurement - self.prev_measurement) / dt

        # Low-pass filter the raw derivative estimate.
        if self.tau_d == 0.0:
            self.derivative = d_meas_dt
        else:
            self.derivative = low_pass_filter(
                self.derivative,
                d_meas_dt,
                dt,
                self.tau_d,
            )

        # Tentative integral update.
        integral_candidate = self.integral + error * dt

        # Unsaturated controller output.
        u_unsat = (
            self.u_bias
            + self.Kp * error
            + self.Ki * integral_candidate
            - self.Kd * self.derivative
        )
        self.u_unsat = u_unsat

        # First apply hard output saturation.
        u_sat = saturation(u_unsat, self.u_min, self.u_max)
        self.u_sat = u_sat

        # Then optionally apply slew-rate limiting to the final output.
        if self.du_dt_limit is None or self.prev_u is None:
            u = u_sat
        else:
            u = rate_limit(
                prev_value=self.prev_u,
                target_value=u_sat,
                dt=dt,
                max_rate=self.du_dt_limit,
            )

        self.u = u

        # Freeze integral if the unsaturated output is beyond the allowable range
        # and the current error would push farther into that same saturation limit.
        pushing_further_high = (u_unsat > self.u_max) and (error > 0.0)
        pushing_further_low = (u_unsat < self.u_min) and (error < 0.0)

        if not (pushing_further_high or pushing_further_low):
            self.integral = integral_candidate

        self.prev_measurement = measurement
        self.prev_u = u

        return u, self.error, self.integral, self.derivative