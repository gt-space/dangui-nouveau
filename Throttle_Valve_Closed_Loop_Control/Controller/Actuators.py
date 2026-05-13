"""
Actuator utilities and simple actuator models.

The actuator is useful when a controller computes a desired command, but the
physical device cannot instantaneously jump to that command. Instead, the
actuator:

1. Enforces hard minimum and maximum output values.
2. Enforces a maximum rate of change per second.

This makes closed-loop simulations more realistic than directly writing the
controller command into the plant.
"""

from .Filters import saturation
from typing import Optional

class FirstOrderActuator:
    """
    Simple rate-limited actuator with hard output bounds.

    This class represents a first-order-ish actuator model for situations where
    a commanded value cannot be achieved instantaneously. The actuator stores
    an internal state `value`, representing the actual physical actuator output.

    At each update step, the actuator:

    1. Saturates the incoming command to the allowable position range.
    2. Computes the difference between the command and the current actuator value.
    3. Limits that difference so the actuator cannot move faster than `max_rate`.
    4. Updates the internal actuator value.
    5. Saturates the final value again to guarantee it remains in bounds.

    This is especially useful in control simulations where the controller
    outputs a desired command, but the plant should receive a more realistic
    actuator response.

    Parameters
    ----------
    initial_value : float
        Initial actuator output value.
    min_value : float
        Minimum allowable actuator output.
    max_value : float
        Maximum allowable actuator output.
    max_rate : float
        Maximum actuator rate, in output-units per second.
        For a valve CdA actuator, this would typically be in m^2/s.

    Attributes
    ----------
    value : float
        Current actuator output value.
    min_value : float
        Minimum allowable actuator output.
    max_value : float
        Maximum allowable actuator output.
    max_rate : float
        Maximum actuator slew rate.

    Raises
    ------
    ValueError
        If:
        - min_value > max_value
        - max_rate < 0
        - initial_value is outside [min_value, max_value]

    Notes
    -----
    If the controller command is already within the actuator's achievable
    change for the current timestep, the actuator will exactly reach the command.

    If the command is too far away, the actuator will move only by the maximum
    allowed increment:

        max_step = max_rate * dt

    so that the actuator respects its finite rate limit.

    Mathematical Form
    -----------------
    Let:

    - u_cmd be the commanded actuator value
    - u_k be the current actuator value
    - dt be the timestep
    - r_max be the maximum rate

    Then the update law is:

        u_cmd_sat = sat(u_cmd, u_min, u_max)

        max_step = r_max * dt

        delta = sat(u_cmd_sat - u_k, -max_step, +max_step)

        u_{k+1} = sat(u_k + delta, u_min, u_max)

    This produces bounded, rate-limited motion toward the commanded value.

    Examples
    --------
    >>> actuator = TestActuator(
    ...     initial_value=0.5e-4,
    ...     min_value=0.2e-4,
    ...     max_value=1.0e-4,
    ...     max_rate=1.0e-5,
    ... )
    >>> actuator.update(command=0.9e-4, dt=0.01)
    5.01e-05

    In that example, the actuator does not jump directly to 0.9e-4. Instead,
    it moves only by the maximum amount allowed during that timestep.
    """

    def __init__(self, initial_value : float, min_value : float, max_value : float, max_rate: Optional[float] = None):
        min_value = float(min_value)
        max_value = float(max_value)
        initial_value = float(initial_value)

        if min_value > max_value:
            raise ValueError(
                f"min_value must be <= max_value. Got min_value={min_value}, max_value={max_value}."
            )

        if max_rate < 0.0:
            raise ValueError(f"max_rate must be nonnegative. Got max_rate={max_rate}.")

        if not (min_value <= initial_value <= max_value):
            raise ValueError(
                "initial_value must lie within [min_value, max_value]. "
                f"Got initial_value={initial_value}, min_value={min_value}, max_value={max_value}."
            )

        self.value = initial_value
        self.min_value = min_value
        self.max_value = max_value
        self.max_rate = max_rate

    def update(self, command, dt):
        """
        Advance the actuator by one timestep toward the commanded value.

        Parameters
        ----------
        command : float
            Desired actuator value from the controller.
        dt : float
            Simulation timestep in seconds. Must be nonnegative.

        Returns
        -------
        float
            Updated actuator output value after applying:
            - command saturation
            - slew-rate limiting
            - final output saturation

        Raises
        ------
        ValueError
            If dt < 0.

        Notes
        -----
        The actuator does not necessarily reach the commanded value in one
        timestep. Instead, it moves toward the command no faster than:

            max_rate * dt

        per update call.

        Behavior summary:
        - If `command` exceeds position limits, it is clipped.
        - If the required motion exceeds the allowed rate, it is reduced.
        - The internal actuator state `self.value` is updated and returned.
        """
        dt = float(dt)
        command = float(command)

        if dt < 0.0:
            raise ValueError(f"dt must be nonnegative. Got dt={dt}.")

        command = saturation(command, self.min_value, self.max_value)

        if self.max_rate is None:
            self.value = command
        else:
            max_step = self.max_rate * dt
            delta = command - self.value
            delta = saturation(delta, -max_step, max_step)

            self.value += delta
            self.value = saturation(self.value, self.min_value, self.max_value)

        return self.value

    def reset(self, value=None):
        if value is None:
            value = self.min_value

        value = float(value)

        if not (self.min_value <= value <= self.max_value):
            raise ValueError(
                "Reset value must lie within [min_value, max_value]. "
                f"Got value={value}, min_value={self.min_value}, max_value={self.max_value}."
            )

        self.value = value


    def reset(self, value=None):
        """
        Reset the actuator state.

        Parameters
        ----------
        value : float or None, optional
            Value to reset the actuator to. If None, the actuator resets to
            `min_value`.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the provided reset value lies outside the actuator bounds.
        """
        if value is None:
            value = self.min_value

        value = float(value)

        if not (self.min_value <= value <= self.max_value):
            raise ValueError(
                "Reset value must lie within [min_value, max_value]. "
                f"Got value={value}, min_value={self.min_value}, max_value={self.max_value}."
            )

        self.value = value

    @property
    def normalized_value(self):
        """
        Return actuator position normalized to [0, 1].

        Returns
        -------
        float
            Normalized actuator value:
            - 0 corresponds to min_value
            - 1 corresponds to max_value

        Notes
        -----
        This can be useful for plotting actuator travel as a fraction of full
        stroke instead of in physical units.
        """
        span = self.max_value - self.min_value
        if span == 0.0:
            return 0.0
        return (self.value - self.min_value) / span

    def __repr__(self):
        """
        Return a developer-friendly string representation of the actuator.
        """
        return (
            f"{self.__class__.__name__}("
            f"value={self.value}, "
            f"min_value={self.min_value}, "
            f"max_value={self.max_value}, "
            f"max_rate={self.max_rate})"
        )
    




class SecondOrderActuator:
    """
    Second-order actuator with hard output bounds, optional velocity limit,
    and hard-stop handling.

    This actuator is useful when a controller computes a desired command, but
    the physical device behaves more like a damped dynamic system than an ideal
    rate-limited ramp. Compared to a simple first-order / slew-limited actuator,
    this model can capture:

    1. finite response speed
    2. actuator inertia
    3. damping
    4. overshoot / ringing (if underdamped)
    5. gradual settling to the command

    States
    ------
    value : float
        Current actuator output value.
    velocity : float
        Current actuator velocity, i.e. d(value)/dt.

    Continuous-Time Model
    ---------------------
    Let:
        x = actuator output
        v = actuator velocity
        u = commanded actuator output

    Then the actuator dynamics are modeled as:

        dx/dt = v

        dv/dt = wn^2 * (u - x) - 2 * zeta * wn * v

    where:
        wn   = natural frequency [rad/s]
        zeta = damping ratio [-]

    Interpretation
    --------------
    - The term wn^2 * (u - x) pulls the actuator toward the command.
    - The term -2 * zeta * wn * v damps the motion.
    - Larger wn gives a faster actuator.
    - Smaller zeta gives more oscillatory / underdamped behavior.
    - zeta = 1.0 is critically damped.
    - zeta > 1.0 is overdamped.
    - 0 < zeta < 1.0 is underdamped.

    Parameters
    ----------
    initial_value : float
        Initial actuator output value.
    min_value : float
        Minimum allowable actuator output.
    max_value : float
        Maximum allowable actuator output.
    wn : float
        Natural frequency [rad/s]. Must be > 0.
    zeta : float
        Damping ratio [-]. Must be >= 0.
    initial_velocity : float, optional
        Initial actuator velocity. Default is 0.0.
    max_velocity : float or None, optional
        Optional hard limit on actuator velocity in output-units per second.
        If None, no explicit velocity clipping is applied.

    Attributes
    ----------
    value : float
        Current actuator output value.
    velocity : float
        Current actuator velocity.
    min_value : float
        Minimum allowable actuator output.
    max_value : float
        Maximum allowable actuator output.
    wn : float
        Natural frequency [rad/s].
    zeta : float
        Damping ratio [-].
    max_velocity : float or None
        Optional hard velocity limit.

    Notes
    -----
    This class uses simple forward Euler integration:

        x_{k+1} = x_k + v_k * dt
        v_{k+1} = v_k + a_k * dt

    where:

        a_k = wn^2 * (u_k - x_k) - 2 * zeta * wn * v_k

    The command is first saturated to [min_value, max_value].

    Hard-stop handling is included:
    - if the actuator hits min_value and is still moving lower,
      velocity is zeroed
    - if the actuator hits max_value and is still moving higher,
      velocity is zeroed

    This prevents the actuator from numerically "pushing through" hard stops.

    Raises
    ------
    ValueError
        If:
        - min_value > max_value
        - initial_value is outside [min_value, max_value]
        - wn <= 0
        - zeta < 0
        - max_velocity is not None and <= 0
    """

    def __init__(
        self,
        initial_value,
        min_value,
        max_value,
        wn,
        zeta,
        initial_velocity=0.0,
        max_velocity=None,
    ):
        min_value = float(min_value)
        max_value = float(max_value)
        initial_value = float(initial_value)
        wn = float(wn)
        zeta = float(zeta)
        initial_velocity = float(initial_velocity)

        if min_value > max_value:
            raise ValueError(
                f"min_value must be <= max_value. Got min_value={min_value}, max_value={max_value}."
            )

        if not (min_value <= initial_value <= max_value):
            raise ValueError(
                "initial_value must lie within [min_value, max_value]. "
                f"Got initial_value={initial_value}, min_value={min_value}, max_value={max_value}."
            )

        if wn <= 0.0:
            raise ValueError(f"wn must be > 0. Got wn={wn}.")

        if zeta < 0.0:
            raise ValueError(f"zeta must be >= 0. Got zeta={zeta}.")

        if max_velocity is not None:
            max_velocity = float(max_velocity)
            if max_velocity <= 0.0:
                raise ValueError(
                    f"max_velocity must be > 0 when provided. Got max_velocity={max_velocity}."
                )

        self.value = initial_value
        self.velocity = initial_velocity

        self.min_value = min_value
        self.max_value = max_value

        self.wn = wn
        self.zeta = zeta
        self.max_velocity = max_velocity

    def update(self, command, dt):
        """
        Advance the actuator by one timestep.

        Parameters
        ----------
        command : float
            Desired actuator output from the controller.
        dt : float
            Simulation timestep in seconds. Must be positive.

        Returns
        -------
        float
            Updated actuator output value after applying:
            - command saturation
            - second-order dynamics
            - optional velocity saturation
            - hard-stop handling

        Raises
        ------
        ValueError
            If dt <= 0.

        Notes
        -----
        Update sequence:
        1. Saturate command into [min_value, max_value]
        2. Compute acceleration from second-order dynamics
        3. Integrate velocity forward by dt
        4. Optionally clip velocity to +/- max_velocity
        5. Integrate position forward by dt
        6. Enforce hard stops and zero velocity if pushing into a stop
        """
        dt = float(dt)
        command = float(command)

        if dt <= 0.0:
            raise ValueError(f"dt must be positive. Got dt={dt}.")

        # Saturate command to allowable actuator travel range
        command = saturation(command, self.min_value, self.max_value)

        # Second-order acceleration:
        #   value_ddot = wn^2 * (command - value) - 2*zeta*wn*velocity
        acceleration = (
            self.wn**2 * (command - self.value)
            - 2.0 * self.zeta * self.wn * self.velocity
        )

        # Integrate velocity
        self.velocity += acceleration * dt

        # Optional explicit velocity limit
        if self.max_velocity is not None:
            self.velocity = saturation(
                self.velocity,
                -self.max_velocity,
                self.max_velocity,
            )

        # Integrate position
        self.value += self.velocity * dt

        # Hard-stop handling
        if self.value <= self.min_value:
            self.value = self.min_value
            if self.velocity < 0.0:
                self.velocity = 0.0

        elif self.value >= self.max_value:
            self.value = self.max_value
            if self.velocity > 0.0:
                self.velocity = 0.0

        return self.value

    def reset(self, value=None, velocity=0.0):
        """
        Reset the actuator state.

        Parameters
        ----------
        value : float or None, optional
            Value to reset the actuator to. If None, resets to min_value.
        velocity : float, optional
            Velocity to reset the actuator to. Default is 0.0.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the provided reset value lies outside the actuator bounds.
        """
        if value is None:
            value = self.min_value

        value = float(value)
        velocity = float(velocity)

        if not (self.min_value <= value <= self.max_value):
            raise ValueError(
                "Reset value must lie within [min_value, max_value]. "
                f"Got value={value}, min_value={self.min_value}, max_value={self.max_value}."
            )

        self.value = value
        self.velocity = velocity

    @property
    def normalized_value(self):
        """
        Return actuator position normalized to [0, 1].

        Returns
        -------
        float
            Normalized actuator value:
            - 0 corresponds to min_value
            - 1 corresponds to max_value
        """
        span = self.max_value - self.min_value
        if span == 0.0:
            return 0.0
        return (self.value - self.min_value) / span

    def __repr__(self):
        """
        Return a developer-friendly string representation of the actuator.
        """
        return (
            f"{self.__class__.__name__}("
            f"value={self.value}, "
            f"velocity={self.velocity}, "
            f"min_value={self.min_value}, "
            f"max_value={self.max_value}, "
            f"wn={self.wn}, "
            f"zeta={self.zeta}, "
            f"max_velocity={self.max_velocity})"
        )