from __future__ import annotations

import numpy as np
import copy
from functools import lru_cache
from CoolProp.CoolProp import PropsSI
from scipy.optimize import toms748, root
import warnings
from .FLUID_NAMES import FLUID_NAME_BANK


def normalize_fluid_name(fluid_name: str) -> str:
    """
    Convert a user-provided fluid name or alias into a CoolProp-compatible
    pure fluid name.

    The input is lowercased and stripped before lookup.

    Parameters
    ----------
    fluid_name : str
        User-provided fluid name.

    Returns
    -------
    str
        CoolProp-compatible fluid name.

    Raises
    ------
    TypeError
        If fluid_name is not a string.
    ValueError
        If fluid_name is empty.
    KeyError
        If no matching CoolProp fluid name or alias is found.
    """
    if not isinstance(fluid_name, str):
        raise TypeError("fluid_name must be a string.")

    key = fluid_name.strip().lower()

    if not key:
        raise ValueError("fluid_name cannot be empty.")

    # 🔑 Lookup in alias bank
    if key in FLUID_NAME_BANK:
        return FLUID_NAME_BANK[key]

    # 🔁 Fallback: try direct CoolProp name
    try:
        PropsSI("D", "P", 101325.0, "T", 300.0, fluid_name)
        return fluid_name
    except Exception:
        pass

    raise KeyError(
        f"Unknown fluid '{fluid_name}'. Add it to fluid_name_bank.py if needed."
    )


@lru_cache(maxsize=None)
def _normalized_fluid_name(fluid_name: str) -> str:
    """
    Cached normalized fluid name.
    """
    return normalize_fluid_name(fluid_name)


@lru_cache(maxsize=None)
def _critical_properties(coolprop_name: str) -> tuple[float, float]:
    """
    Cached critical temperature and pressure.

    Returns
    -------
    tuple[float, float]
        (T_crit [K], P_crit [Pa])
    """
    return (
        PropsSI("Tcrit", coolprop_name),
        PropsSI("Pcrit", coolprop_name),
    )


@lru_cache(maxsize=None)
def _saturation_properties(coolprop_name: str, temperature: float) -> tuple[float, float, float]:
    """
    Cached saturation properties at a given temperature.

    Returns
    -------
    tuple[float, float, float]
        (P_sat [Pa], rho_sat_liq [kg/m^3], rho_sat_vap [kg/m^3])
    """
    return (
        PropsSI("P", "T", temperature, "Q", 0, coolprop_name),
        PropsSI("D", "T", temperature, "Q", 0, coolprop_name),
        PropsSI("D", "T", temperature, "Q", 1, coolprop_name),
    )


def get_density(
    fluid_name: str,
    pressure: float,
    temperature: float,
    quality: float | None = 0.0,
    sat_rtol: float = 1e-6,
    verbose: bool = False,
) -> float:
    """
    Return the density of a fluid using CoolProp, after normalizing common
    alternate names to CoolProp-compatible pure fluid names.

    For pure fluids, density is not uniquely determined by (P, T) exactly on the
    saturation line. In that case, this function uses the provided quality factor.

    Parameters
    ----------
    fluid_name : str
        Fluid name or alias (e.g. 'Water', 'lox', 'RP-1', 'kerosene').
    pressure : float
        Pressure in Pa.
    temperature : float
        Temperature in K.
    quality : float | None, optional
        Vapor quality used when (P, T) lies on the saturation line.

        - 0.0 -> saturated liquid density
        - 1.0 -> saturated vapor density
        - 0 < quality < 1 -> two-phase mixture density
        - None -> do not resolve saturation ambiguity; call PropsSI directly

        Default is 0.0.
    sat_rtol : float, optional
        Relative tolerance used to determine whether the given pressure is at
        saturation for the specified temperature.
    verbose : bool, optional
        If True, emit warnings/messages for saturation handling.

    Returns
    -------
    float
        Density in kg/m^3.

    Raises
    ------
    ValueError
        If inputs are invalid or the quality is outside [0, 1].
    """
    if pressure <= 0.0:
        raise ValueError("pressure must be positive.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if quality is not None and not (0.0 <= quality <= 1.0):
        raise ValueError("quality must be between 0 and 1, or None.")

    coolprop_name = _normalized_fluid_name(fluid_name)
    T_crit, _ = _critical_properties(coolprop_name)

    if temperature < T_crit and quality is not None:
        try:
            P_sat, _, _ = _saturation_properties(coolprop_name, temperature)
            scale = max(abs(P_sat), abs(pressure), 1.0)

            if abs(pressure - P_sat) <= sat_rtol * scale:
                if verbose:
                    warnings.warn(
                        "State lies on the saturation line; using the provided quality.",
                        RuntimeWarning,
                    )
                return PropsSI("D", "P", pressure, "Q", quality, coolprop_name)
        except Exception:
            pass

    return PropsSI("D", "P", pressure, "T", temperature, coolprop_name)


def get_pressure(
    fluid_name: str,
    density: float,
    temperature: float,
    tol: float = 1e-6,
    density_rtol: float = 1e-6,
    verbose: bool = False,
) -> float:
    """
    Return pressure [Pa] from density [kg/m^3] and temperature [K] for a pure fluid
    using CoolProp, with explicit handling of saturation and critical behavior.

    This function is intended for pure-fluid states where density and temperature
    are known and pressure must be recovered numerically. For subcritical
    temperatures, it explicitly checks the saturation state at the specified
    temperature and handles the following cases:

    - density equal to saturated liquid density -> return saturation pressure
    - density equal to saturated vapor density -> return saturation pressure
    - density between saturated vapor and saturated liquid density -> interpret
      the state as two-phase and return saturation pressure
    - density below saturated vapor density -> solve on the vapor branch
    - density above saturated liquid density -> solve on the compressed-liquid branch

    For critical and supercritical temperatures, the function solves directly on
    a single branch using a bracketed root finder.

    A small pressure offset is applied when solving just above or below the
    saturation line so that CoolProp is not queried exactly at saturation,
    which can otherwise raise errors for (P, T) inputs that are numerically too
    close to the coexistence boundary.

    Parameters
    ----------
    fluid_name : str
        Fluid name or alias.
    density : float
        Density in kg/m^3.
    temperature : float
        Temperature in K.
    tol : float, optional
        Absolute pressure tolerance passed to the root solver.
    density_rtol : float, optional
        Relative tolerance used for saturation-density comparisons.
    verbose : bool, optional
        If True, emit warnings/messages for two-phase handling.

    Returns
    -------
    float
        Pressure in Pa.

    Raises
    ------
    ValueError
        If inputs are invalid or no physically valid pressure can be found.
    """
    if density <= 0.0:
        raise ValueError("density must be positive.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")

    coolprop_name = _normalized_fluid_name(fluid_name)
    T_crit, P_crit = _critical_properties(coolprop_name)

    # Margin used to move off the saturation line before querying CoolProp
    # on the vapor or compressed-liquid branches.
    sat_pressure_margin = 1e-5

    def rho_from_PT(P: float) -> float:
        return PropsSI("D", "P", P, "T", temperature, coolprop_name)

    def residual(P: float) -> float:
        return rho_from_PT(P) - density

    def is_close(a: float, b: float, rtol: float = density_rtol) -> bool:
        scale = max(abs(a), abs(b), 1.0)
        return abs(a - b) <= rtol * scale

    # ---------------------------------------------------------
    # 1) Subcritical temperature: explicitly handle saturation
    # ---------------------------------------------------------
    if temperature < T_crit:
        try:
            P_sat, rho_sat_liq, rho_sat_vap = _saturation_properties(
                coolprop_name,
                temperature,
            )

            if is_close(density, rho_sat_liq):
                return P_sat

            if is_close(density, rho_sat_vap):
                return P_sat

            if rho_sat_vap < density < rho_sat_liq:
                if verbose:
                    warnings.warn(
                        "Requested (rho, T) lies in the two-phase region for a pure fluid. "
                        "Returning saturation pressure. Density alone does not determine quality.",
                        RuntimeWarning,
                    )
                return P_sat

            # Vapor branch
            if density < rho_sat_vap:
                P_low = 1.0
                P_high = P_sat * (1.0 - sat_pressure_margin)

                f_low = residual(P_low)
                f_high = residual(P_high)

                if f_low * f_high > 0:
                    P_low = 1e-6
                    f_low = residual(P_low)

                if f_low * f_high > 0:
                    raise ValueError(
                        f"Could not bracket vapor-phase pressure root for "
                        f"density={density} kg/m^3 at T={temperature} K."
                    )

                return toms748(residual, P_low, P_high, xtol=tol)

            # Compressed-liquid branch
            P_low = P_sat * (1.0 + sat_pressure_margin)
            P_high = max(10.0 * P_crit, 10.0 * P_sat)

            f_low = residual(P_low)
            f_high = residual(P_high)

            n_expand = 0
            while f_low * f_high > 0 and n_expand < 20:
                P_high *= 2.0
                f_high = residual(P_high)
                n_expand += 1

            if f_low * f_high > 0:
                raise ValueError(
                    f"Could not bracket liquid-phase pressure root for "
                    f"density={density} kg/m^3 at T={temperature} K."
                )

            return toms748(residual, P_low, P_high, xtol=tol)

        except ValueError:
            raise
        except Exception as e:
            raise ValueError(
                f"Failed while handling subcritical saturation behavior for "
                f"{fluid_name} at T={temperature} K: {e}"
            )

    # ---------------------------------------------------------
    # 2) Critical or supercritical temperature
    # ---------------------------------------------------------
    P_low = 1.0
    P_high = max(10.0 * P_crit, 1e7)

    try:
        f_low = residual(P_low)
        f_high = residual(P_high)
    except Exception as e:
        raise ValueError(
            f"Failed evaluating EOS for {fluid_name} at T={temperature} K: {e}"
        )

    n_expand = 0
    while f_low * f_high > 0 and n_expand < 25:
        P_high *= 2.0
        f_high = residual(P_high)
        n_expand += 1

    if f_low * f_high > 0:
        raise ValueError(
            f"Could not bracket pressure root for density={density} kg/m^3 "
            f"at T={temperature} K (critical/supercritical region)."
        )

    return toms748(residual, P_low, P_high, xtol=tol)



def incompressible_CdA_equation(P1: float, P2: float,   # Pa
                                rho: float,             # kg/m^3
                                CdA: float,             # m^2
                                ) -> float:
    """
    Calculates mass flow for incompressible flow using the CdA equation.

    Parameters
    ----------
    P1 : float
        Upstream pressure [Pa].
    P2 : float
        Downstream pressure [Pa].
    rho : float
        Density [kg/m^3].
    CdA : float
        Effective flow area [m^2].

    Returns
    -------
    float
        Mass flow rate [kg/s].
    """
    return np.sign(P1 - P2) * CdA * np.sqrt(2 * rho * np.abs(P1 - P2))



def series_CdA(*CdAs):
    """
    Return the equivalent series CdA for any number of flow elements in series.

    For incompressible-style pressure loss written in terms of CdA, series elements
    combine as:

        1 / CdA_eq^2 = sum(1 / CdA_i^2)

    Parameters
    ----------
    *CdAs : float
        Any number of positive CdA values [m^2].

    Returns
    -------
    float
        Equivalent series CdA [m^2].

    Raises
    ------
    ValueError
        If no CdAs are provided or any CdA is nonpositive.
    """
    if len(CdAs) == 0:
        raise ValueError("At least one CdA must be provided.")

    if len(CdAs) == 1 and isinstance(CdAs[0], (list, tuple, np.ndarray)):
        CdAs = CdAs[0]

    CdAs = np.asarray(CdAs, dtype=float)

    if CdAs.size == 0:
        raise ValueError("At least one CdA must be provided.")

    if np.any(CdAs <= 0):
        raise ValueError("All CdA values must be positive.")

    return np.sqrt(1.0 / np.sum(1.0 / CdAs**2))






def solve_system_CdAs(
    ts,
    *,
    Pc_target: float,
    MR_target: float,
    x0_CdA: tuple[float, float] | None = None,
    tol: float = 1e-9,
    maxfev: int = 250,
    CdA_min: float = 1e-10,
    CdA_max: float = 1e-1,
    Pc_scale: float | None = None,
    MR_scale: float | None = None,
    verbose: bool = False,
):
    """
    Solve for system throttle CdAs that achieve a target chamber pressure and mixture ratio.

    This routine determines the effective discharge areas of the fuel and oxidizer
    **system throttles** such that the steady-state solution of the TestStand satisfies:

        Pc(ts) ≈ Pc_target
        MR(ts) ≈ MR_target

    where Pc is the combustion chamber pressure and MR is the oxidizer-to-fuel
    mass mixture ratio.

    The solution is obtained by wrapping the TestStand steady-state solver
    inside a nonlinear root-finding problem using SciPy's `root(method="hybr")`.

    The unknowns are:

        CdA_fuel_system
        CdA_oxidizer_system

    which correspond to:

        ts.FuelThrottleValve.CdA
        ts.OxThrottleValve.CdA


    --------------------------------------------------------------------------
    Solver Strategy
    --------------------------------------------------------------------------

    The root solver operates on a transformed variable:

        u = log(CdA)

    so that:

        CdA = exp(u)

    This guarantees that CdA remains strictly positive during iteration,
    preventing the solver from exploring non-physical negative flow areas.

    During each residual evaluation:

    1. The CdAs are reconstructed from the log variables.
    2. They are clamped to `[CdA_min, CdA_max]` to avoid pathological values.
    3. The throttle CdAs of a working copy of the TestStand are updated.
    4. The TestStand steady-state solver is executed.
    5. Residuals are computed as normalized errors:

            r1 = (Pc - Pc_target) / Pc_scale
            r2 = (MR - MR_target) / MR_scale

    These residuals are returned to the root solver.


    --------------------------------------------------------------------------
    Warm-Starting the Inner Solver
    --------------------------------------------------------------------------

    The steady-state solver itself performs a nonlinear solve for internal
    pressures (injector manifold pressures and chamber pressure).

    To accelerate convergence, this routine **warm-starts** each new
    steady-state solve using the previous successful solution:

        last_x0 = solved.solved_x0()

    This dramatically reduces the number of inner solver iterations when
    the outer CdA iteration makes only small changes.


    --------------------------------------------------------------------------
    Copy Semantics
    --------------------------------------------------------------------------

    The original TestStand object provided by the caller is **never mutated**.

    Instead:

        base = deepcopy(ts)

    is created once at the start of the solve. Only this internal working copy
    has its CdAs modified during the iteration.

    After convergence, a **fresh deep copy of the caller's TestStand** is
    created and the solved CdAs are applied before running one final
    steady-state solve to produce the returned result.

    A custom `TestStand.__deepcopy__` implementation ensures that the
    RocketCEA object (`_cea_obj`) is shared rather than copied, avoiding
    expensive duplication of thermochemistry objects.


    --------------------------------------------------------------------------
    Failure Handling
    --------------------------------------------------------------------------

    If the steady-state solver fails at a candidate CdA pair, the routine
    does **not raise immediately**. Instead it returns a large penalty
    residual to the root solver so that iteration may continue.

    This prevents occasional infeasible points from terminating the solve.


    --------------------------------------------------------------------------
    Parameters
    --------------------------------------------------------------------------
    ts : TestStand
        Baseline test stand configuration. This object is treated as
        immutable and will not be modified by the solver.

    Pc_target : float
        Desired combustion chamber pressure [Pa].

    MR_target : float
        Desired oxidizer-to-fuel mass mixture ratio.

    x0_CdA : tuple[float, float] | None, optional
        Initial guess for the system CdAs:

            (FuelThrottleValve.CdA, OxThrottleValve.CdA)

        If None, the values currently stored in the TestStand are used.

    tol : float, optional
        Root-finding convergence tolerance passed to SciPy's `root`
        (`xtol` parameter).

    maxfev : int, optional
        Maximum number of residual evaluations allowed by the root solver.

    CdA_min : float, optional
        Lower bound applied to CdA during iteration to prevent degenerate
        flow elements.

    CdA_max : float, optional
        Upper bound applied to CdA to prevent unrealistically large flow areas.

    Pc_scale : float | None, optional
        Scaling factor used when forming the chamber pressure residual.

        If None, it defaults to:

            max(abs(Pc_target), 1)

        which keeps the residual magnitude O(1).

    MR_scale : float | None, optional
        Scaling factor used when forming the mixture ratio residual.

    verbose : bool, optional
        If True, prints iteration diagnostics including CdA guesses,
        resulting chamber pressure, mixture ratio, and residual values.


    --------------------------------------------------------------------------
    Returns
    --------------------------------------------------------------------------
    tuple
        A tuple containing:

        CdA_fuel : float
            Solved fuel throttle CdA [m²]

        CdA_ox : float
            Solved oxidizer throttle CdA [m²]

        solved_teststand : TestStand
            A fully solved TestStand instance containing the converged
            steady-state operating point.


    --------------------------------------------------------------------------
    Raises
    --------------------------------------------------------------------------
    ValueError
        If the initial CdA guess is non-positive.

    RuntimeError
        If the root solver fails to converge.


    --------------------------------------------------------------------------
    Notes
    --------------------------------------------------------------------------

    • This routine is most useful when designing feed systems where the
      system throttles must be sized to achieve a desired operating point.

    • Because each residual evaluation requires a steady-state solve, the
      computational cost is dominated by the inner TestStand solver.

    • The global RocketCEA object cache and the custom TestStand deepcopy
      implementation ensure that thermochemistry objects are **not recreated
      or copied during the solve**, keeping the iteration efficient.

    """

    base = copy.deepcopy(ts)  # safe sandbox (won't recreate CEA if you cache + __deepcopy__)
    last_x0 = base.get_x0()

    if x0_CdA is None:
        x0_CdA = (float(base.FuelThrottleValve.CdA), float(base.OxThrottleValve.CdA))

    CdA_f0, CdA_ox0 = map(float, x0_CdA)
    if CdA_f0 <= 0 or CdA_ox0 <= 0:
        raise ValueError("Initial CdA guesses must be > 0.")

    # scaling to keep residuals balanced
    if Pc_scale is None:
        Pc_scale = max(abs(Pc_target), 1.0)
    if MR_scale is None:
        MR_scale = max(abs(MR_target), 1.0)

    BIG = 1e9

    def clamp(x: float) -> float:
        return float(np.clip(x, CdA_min, CdA_max))

    # log-transform => always positive
    u0 = np.log([CdA_f0, CdA_ox0])

    def residual_u(u: np.ndarray) -> np.ndarray:
        nonlocal last_x0

        CdA_f = clamp(np.exp(float(u[0])))
        CdA_ox = clamp(np.exp(float(u[1])))

        # mutate only CdAs on base
        base.FuelThrottleValve.CdA = CdA_f
        base.OxThrottleValve.CdA   = CdA_ox

        try:
            solved = base.steady_state(x0=last_x0)
            last_x0 = solved.solved_x0()

            Pc = float(solved.MainChamber.p)
            MR = float(getattr(solved.MainChamber, "MR", np.nan))
            if not np.isfinite(Pc) or not np.isfinite(MR) or Pc <= 0 or MR <= 0:
                return np.array([BIG, BIG], dtype=float)

            rPc = (Pc - Pc_target) / Pc_scale
            rMR = (MR - MR_target) / MR_scale

            if verbose:
                print(f"CdA_f={CdA_f:.3e}, CdA_ox={CdA_ox:.3e} -> Pc={Pc:.3e}, MR={MR:.4f}, r=[{rPc:.3e}, {rMR:.3e}]")

            return np.array([rPc, rMR], dtype=float)

        except Exception as e:
            if verbose:
                print(f"steady_state failed at CdA_f={CdA_f:.3e}, CdA_ox={CdA_ox:.3e}: {e}")
            return np.array([BIG, BIG], dtype=float)

    sol = root(residual_u, x0=u0, method="hybr", options={"xtol": tol, "maxfev": int(maxfev)})

    if not sol.success:
        CdA_f_best = clamp(np.exp(float(sol.x[0])))
        CdA_ox_best = clamp(np.exp(float(sol.x[1])))
        raise RuntimeError(
            "solve_system_CdAs failed.\n"
            f"message: {sol.message}\n"
            f"CdA best guess: fuel={CdA_f_best:.6e}, ox={CdA_ox_best:.6e}"
        )

    CdA_f_sol = clamp(np.exp(float(sol.x[0])))
    CdA_ox_sol = clamp(np.exp(float(sol.x[1])))

    # apply to a fresh copy of the caller and return a final solved state
    out = copy.deepcopy(ts)
    out.FuelThrottleValve.CdA = CdA_f_sol
    out.OxThrottleValve.CdA   = CdA_ox_sol
    out_solved = out.steady_state()

    return CdA_f_sol, CdA_ox_sol, out_solved