from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .TestStand import TestStand



# Helper Functions:

def _split_path(path: str) -> tuple[str, str]:
    """
    Split a dotted attribute path of the form "Object.attr"
    into ("Object", "attr").

    Raises ValueError if the path is not exactly two levels.
    """
    parts = path.split(".")
    if len(parts) != 2:
        raise ValueError(
            f"Expected path format 'Object.attr', got {path!r}"
        )
    return parts[0], parts[1]


def _make_tune_set(path: str) -> Callable[["TestStand", float], None]:
    """
    Build a setter function for a tuning knob.

    Given a path "Object.attr", returns a callable:

        setter(teststand, value)

    that sets:
        teststand.Object.attr = float(value)
    """
    obj_name, attr = _split_path(path)

    def _set(ts: "TestStand", value: float) -> None:
        setattr(getattr(ts, obj_name), attr, float(value))

    return _set


def _make_measure_attr(path: str) -> Callable[["TestStand"], float]:
    """
    Build a getter function for a measured quantity.

    Given a path "Object.attr", returns a callable:

        getter(teststand) -> float

    that evaluates and returns:
        float(teststand.Object.attr)
    """
    obj_name, attr = _split_path(path)

    def _get(ts: "TestStand") -> float:
        return float(getattr(getattr(ts, obj_name), attr))

    return _get


# Balance class
@dataclass(frozen=True)
class Balance:
    """
        Balance specification object used by TestStand.solve_with_balance().

        A Balance fully defines a one-dimensional tuning problem of the form:

            "Adjust <tune> until <measure> == target"

        Every field in this class is logically required to define a complete,
        well-posed balance problem. Even though some parameters provide defaults,
        they are conceptually required components of the tuning definition.

        Required Inputs
        ---------------
        tune : str
            REQUIRED.
            Dotted attribute path of the form "Object.attr" identifying the
            parameter to be adjusted.

            Examples:
                "OxThrottleValve.CdA"
                "FuelThrottleValve.CdA"
                "TCA.At"

            This attribute must exist on the TestStand instance being solved.

        measure : str OR callable
            REQUIRED.
            Defines what quantity will be evaluated after each steady-state solve.

            Option 1 (standard usage):
                A dotted attribute path "Object.attr" that exists on the SOLVED
                TestStand instance.

                Examples:
                    "MainChamber.p"
                    "MainChamber.MR"
                    "TCA.F"
                    "FuelInjector.stiffness"

            Option 2 (advanced usage):
                A callable:

                    measure_fn(solved_teststand) -> float

                This allows arbitrary expressions (ratios, percentages, multi-variable
                functions, etc.).

        target : float
            REQUIRED.
            The desired value of the measured quantity.

        bounds : (float, float)
            REQUIRED.
            Lower and upper bounds for the tuning parameter.
            Must satisfy: 0 < lower < upper.

            The solver will search only within this interval.

        tol : float
            REQUIRED.
            Convergence tolerance applied to:

                abs(measure - target)

            Balancing stops when the measured value is within this tolerance.

        name : str
            REQUIRED.
            Descriptive name for the balance. Used for logging, debugging,
            and reporting.

            If not explicitly provided, a name is automatically generated,
            but conceptually every Balance should have a descriptive label.

        Design Notes
        ------------
        • Balance does NOT perform solving.
        It only describes WHAT should be tuned and WHAT condition defines success.

        • Balance is layout-agnostic. It only requires that the provided
        attribute paths exist on the TestStand object passed to
        solve_with_balance().

        Example
        -------
            MR_balance = Balance(
                tune="OxThrottleValve.CdA",
                measure="MainChamber.MR",
                target=2.0,
                bounds=(1e-10, 1e-4),
                tol=1e-5,
                name="MR Control"
            )

        This defines the problem:
            "Adjust OxThrottleValve.CdA until MainChamber.MR equals 2.0"
        """

    tune: str
    measure: Union[str, Callable[["TestStand"], float]]
    target: float
    bounds: Tuple[float, float] = (1e-8, 5e-4)
    tol: float = 1e-6
    name: str | None = None

    def __post_init__(self):
        lo, hi = self.bounds
        if not (lo > 0 and hi > 0 and hi > lo):
            raise ValueError("bounds must be (lo, hi) with 0 < lo < hi.")

        # Build tune_set
        tune_set = _make_tune_set(self.tune)

        # Build measure function
        if callable(self.measure):
            measure_fn = self.measure
            measure_label = getattr(self.measure, "__name__", "custom")
        else:
            m = self.measure.strip()
            if "." not in m:
                raise ValueError(
                    f"measure must be an 'Object.attr' path (e.g., 'MainChamber.p'), got {m!r}"
                )
            measure_fn = _make_measure_attr(m)
            measure_label = m

        # Auto-generate name if not provided
        if self.name is None:
            auto_name = f"Tune {self.tune} until {measure_label} = {self.target}"
            object.__setattr__(self, "name", auto_name)

        # Store compiled versions (private attributes)
        object.__setattr__(self, "_tune_set", tune_set)
        object.__setattr__(self, "_measure_fn", measure_fn)

    @property
    def tune_set(self):
        return self._tune_set

    @property
    def measure_fn(self):
        return self._measure_fn

    def describe(self) -> str:
        lo, hi = self.bounds
        return (
            f"Balance[{self.name}] "
            f"target={self.target} "
            f"knob_bounds=[{lo:.3e}, {hi:.3e}] "
            f"tol={self.tol}"
        )