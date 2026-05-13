from __future__ import annotations

import copy
import numpy as np
from scipy.optimize import root, least_squares
import pandas as pd

from .Components import *
from Utilities import choked_nozzle_thrust, choked_nozzle_mass_flow, get_chamber_pressure
from Utilities import incompressible_CdA_equation, get_density, get_pressure
from Utilities import get_cached_CEA
from Utilities.Constants import *

from .Balance import Balance

class TestStand:
    """
    Lumped-parameter test stand model for steady-state feed + injector + chamber + nozzle coupling.

    This object composes component instances representing the fuel and oxidizer
    flow paths feeding a combustion chamber and a choked nozzle throat.

    Assumptions
    ------------------------------------
    - Incompressible flow through CdA elements for feed system + injectors.
    - No two-phase effects, flashing, cavitation, or temperature coupling.

    Parameters
    ----------
    name: str
        Identifier for test stand.
    FuelTank, OxTank : Tank
        Tanks holding propellants, providing upstream pressure and density.
        Expected attributes used here include: `.p` [Pa], `.rho` [kg/m^3], and `.propellant` (name).
    FuelRunline, OxRunline : Line
        Feed lines. s
    FuelThrottleValve, OxThrottleValve : Valve
        Throttle elements providing system CdA on each side.
        Expected attribute: `.CdA` [m^2].
    FuelInjectorManifold, OxInjectorManifold : InjectorManifold
        Manifolds upstream of injector orifices; used for initial guesses and may store solved pressures.
        Expected attribute: `.p` [Pa].
    FuelInjector, OxInjector : Orifice
        Injector elements modeled as incompressible CdA elements.
        Expected attributes: `.CdA` [m^2]; this solver will also populate `.mdot` [kg/s] if present.
    MainChamber : CombustionChamber
        Combustion chamber model; expected attributes: `.p` [Pa] initial guess and `.eta_cstar` (dimensionless).
    TCA : Nozzle
        Nozzle/throat model; expected attribute: `.At` [m^2].
    Ambient : Drain
        Ambient atmospheric model; expected attribute: `.p` [Pa].

    Notes
    -----
    - This class is designed so you can keep an "unsolved" configuration `A` and generate a solved
      configuration `B` without mutating `A`.
    
    - If, for any reason, the layout of TestStand needs to be altered (whether a new volume node needs
      to be added, or the entire system needs to be reconfigured), because this code is not entirely 
      modular, several keys items must be modified:

      1) Any hardcoded attributes references (e.g. self.FuelTank.p, self.MainChamber.MR) need to be
         modified in the class and the residual function.
    
      2) The number of residuals and iteration variables in the residual function should be altered.
         Currently, len(STEADY_STATE_X0_PATHS) == 3 is used enfore three iteration variables, and
         hence, three residuals. Make sure to alter this guarding code.

      3) STEADY_STATE_X0_PATHS must be updated. The order of the list matters! STEADY_STATE_X0_PATHS
         tells the solver how many variables are iterated on. The order corresponds to the iteration
         variables in the residual function. STEADY_STATE_X0_PATHS can be found above __init__().

      4) __str__ and __repr__ are configuration dependent.

    - Balance.py should agnostic to layout changes, unless the way that component attributes are 
      accessed changes (e.g. FuelSide.Tank.p instead of FuelTank.p)

    - steady_state_with_balance() should also continue to be generic, working for most any layout changes,
      as long as there is a steady_state() function that accepts x0, a get_x0(), and a solved_x0().
      Make sure to also update the residual function and STEADY_STATE_X0_PATHS, and everything should
      be fine.
        
    """


    # Order matters: MUST match the unknown ordering used in residual(x)
    STEADY_STATE_X0_PATHS = [
        "FuelInjectorManifold.p",
        "OxInjectorManifold.p",
        "MainChamber.p",
    ]

    def __init__(self, name: str,
                 FuelTank: Tank, OxTank: Tank,
                 FuelRunline: Line, OxRunline: Line,
                 FuelThrottleValve: Valve, OxThrottleValve: Valve,
                 FuelInjectorManifold: InjectorManifold, OxInjectorManifold: InjectorManifold,
                 FuelInjector: Orifice, OxInjector: Orifice,
                 MainChamber: CombustionChamber, TCA: Nozzle,
                 Ambient: Drain):

        self.name = name
        self.FuelTank = FuelTank
        self.OxTank = OxTank
        self.FuelRunline = FuelRunline
        self.OxRunline = OxRunline
        self.FuelThrottleValve = FuelThrottleValve
        self.OxThrottleValve = OxThrottleValve
        self.FuelInjectorManifold = FuelInjectorManifold
        self.OxInjectorManifold = OxInjectorManifold
        self.FuelInjector = FuelInjector
        self.OxInjector = OxInjector
        self.MainChamber = MainChamber
        self.TCA = TCA
        self.Ambient = Ambient

        fuel_name = self.FuelTank.propellant
        ox_name   = self.OxTank.propellant
        self._cea_obj = get_cached_CEA(fuel_name, ox_name)

    def __repr__(self) -> str:
        return f"TestStand (name={self.name}, FuelTank={self.FuelTank!r}, OxTank={self.OxTank!r}, MainChamber={self.MainChamber!r}, TCA={self.TCA!r})"


    def __str__(self, units: str = "COMBINED") -> str:
        # Local helpers
        def _fmt_fluid(f):
            return f if f is not None else "—"

        def _get(obj, attr):
            v = getattr(obj, attr, None)
            try:
                return float(v) if v is not None else None
            except Exception:
                return None

        def _table(rows, headers):
            """
            rows: list of tuples/iterables of strings
            headers: tuple/list of strings
            """
            cols = list(zip(*([headers] + rows)))
            widths = [max(len(str(c)) for c in col) for col in cols]

            def fmt_row(r):
                return " | ".join(str(c).ljust(w) for c, w in zip(r, widths))

            sep = "-+-".join("-" * w for w in widths)
            out = [fmt_row(headers), sep]
            out += [fmt_row(r) for r in rows]
            return "\n".join(out)

        def _safe_density(fluid_name, pressure, temperature):
            if fluid_name is None or pressure is None or temperature is None:
                return None
            try:
                return get_density(fluid_name, pressure, temperature)
            except Exception:
                return None

        units = (units or "COMBINED").upper()

        if units == "SI":
            def _fmt_pressure(p):
                return f"{p:,.2f} Pa" if p is not None else "—"

            def _fmt_area(a):
                return f"{a:.3e} m^2" if a is not None else "—"

            def _fmt_force(f):
                return f"{f:,.2f} N" if f is not None else "—"

            def _fmt_temperature(T):
                return f"{T:.2f} K" if T is not None else "—"

            def _fmt_density(rho):
                return f"{rho:.1f} kg/m^3" if rho is not None else "—"

            def _fmt_mdot(mdot):
                return f"{mdot:.4f} kg/s" if mdot is not None else "—"

        elif units == "US":
            def _fmt_pressure(p):
                return f"{p / PA_PER_PSI:,.2f} psi" if p is not None else "—"

            def _fmt_area(a):
                return f"{a * IN2_PER_M2:.4f} in^2" if a is not None else "—"

            def _fmt_force(f):
                return f"{f * LBF_PER_N:,.2f} lbf" if f is not None else "—"

            def _fmt_temperature(T):
                return f"{T * RANKINE_PER_KELVIN:.2f} R" if T is not None else "—"

            def _fmt_density(rho):
                return f"{rho * LBM_FT3_PER_KG_M3:.3f} lbm/ft^3" if rho is not None else "—"

            def _fmt_mdot(mdot):
                return f"{mdot * LBM_PER_KG:.4f} lbm/s" if mdot is not None else "—"

        elif units == "COMBINED":
            def _fmt_pressure(p):
                return (
                    f"{p:,.2f} Pa ({p / PA_PER_PSI:,.2f} psi)"
                    if p is not None else "—"
                )

            def _fmt_area(a):
                return (
                    f"{a:.3e} m^2 ({a * IN2_PER_M2:.4f} in^2)"
                    if a is not None else "—"
                )

            def _fmt_force(f):
                return (
                    f"{f:,.2f} N ({f * LBF_PER_N:,.2f} lbf)"
                    if f is not None else "—"
                )

            def _fmt_temperature(T):
                return (
                    f"{T:.2f} K ({T * RANKINE_PER_KELVIN:.2f} R)"
                    if T is not None else "—"
                )

            def _fmt_density(rho):
                return (
                    f"{rho:.1f} kg/m^3 ({rho * LBM_FT3_PER_KG_M3:.3f} lbm/ft^3)"
                    if rho is not None else "—"
                )

            def _fmt_mdot(mdot):
                return (
                    f"{mdot:.4f} kg/s ({mdot * LBM_PER_KG:.4f} lbm/s)"
                    if mdot is not None else "—"
                )

        else:
            raise ValueError("units must be 'SI', 'US', or 'COMBINED'")

        # Pull common numbers
        mdot_f = _get(self.FuelInjector, "mdot")
        mdot_ox = _get(self.OxInjector, "mdot")
        mdot_total = (mdot_f + mdot_ox) if (mdot_f is not None and mdot_ox is not None) else None
        MR = (mdot_ox / mdot_f) if (mdot_f is not None and mdot_f != 0 and mdot_ox is not None) else None

        # Pressures
        P_tank_f = _get(self.FuelTank, "p")
        P_tank_ox = _get(self.OxTank, "p")
        P_inj_f = _get(self.FuelInjectorManifold, "p")
        P_inj_ox = _get(self.OxInjectorManifold, "p")
        Pc = _get(self.MainChamber, "p")
        Pamb = _get(self.Ambient, "p")

        # Temperatures
        T_tank_f = _get(self.FuelTank, "T")
        T_tank_ox = _get(self.OxTank, "T")
        T_inj_f = _get(self.FuelInjectorManifold, "T")
        T_inj_ox = _get(self.OxInjectorManifold, "T")

        # Fluids / propellants
        fuel_propellant = getattr(self.FuelTank, "propellant", None)
        ox_propellant = getattr(self.OxTank, "propellant", None)

        # Densities
        rho_f = _safe_density(fuel_propellant, P_tank_f, T_tank_f)
        rho_ox = _safe_density(ox_propellant, P_tank_ox, T_tank_ox)
        rho_f_man = _safe_density(fuel_propellant, P_inj_f, T_inj_f)
        rho_ox_man = _safe_density(ox_propellant, P_inj_ox, T_inj_ox)

        # Nozzle / thrust / efficiencies
        At = _get(self.TCA, "At")
        eps = _get(self.TCA, "eps")
        F = _get(self.TCA, "F")
        nfz = _get(self.TCA, "nfz")
        eta_cstar = _get(self.MainChamber, "eta_cstar")
        eta_cf = _get(self.TCA, "eta_cf")

        # CdAs
        CdA_f_sys = _get(self.FuelThrottleValve, "CdA")
        CdA_ox_sys = _get(self.OxThrottleValve, "CdA")
        CdA_f_inj = _get(self.FuelInjector, "CdA")
        CdA_ox_inj = _get(self.OxInjector, "CdA")

        # Injector stiffness
        stiff_f = _get(self.FuelInjector, "stiffness")
        stiff_ox = _get(self.OxInjector, "stiffness")

        state_rows = [
            ("Fuel Tank",     _fmt_fluid(fuel_propellant), _fmt_pressure(P_tank_f),  _fmt_temperature(T_tank_f),  _fmt_density(rho_f)),
            ("Fuel Manifold", _fmt_fluid(fuel_propellant), _fmt_pressure(P_inj_f),   _fmt_temperature(T_inj_f),   _fmt_density(rho_f_man)),
            ("Ox Tank",       _fmt_fluid(ox_propellant),   _fmt_pressure(P_tank_ox), _fmt_temperature(T_tank_ox), _fmt_density(rho_ox)),
            ("Ox Manifold",   _fmt_fluid(ox_propellant),   _fmt_pressure(P_inj_ox),  _fmt_temperature(T_inj_ox),  _fmt_density(rho_ox_man)),
            ("Chamber",       "—",                         _fmt_pressure(Pc),         "—",                          "—"),
            ("Ambient",       "—",                         _fmt_pressure(Pamb),       "—",                          "—"),
        ]

        flow_rows = [
            ("Fuel mdot", _fmt_mdot(mdot_f)),
            ("Ox mdot", _fmt_mdot(mdot_ox)),
            ("Total mdot", _fmt_mdot(mdot_total)),
            ("MR", f"{MR:.4f}" if MR is not None else "—"),
            ("Fuel inj stiffness", f"{100.0 * stiff_f:.2f} %" if stiff_f is not None else "—"),
            ("Ox inj stiffness", f"{100.0 * stiff_ox:.2f} %" if stiff_ox is not None else "—"),
        ]

        geom_rows = [
            ("At", _fmt_area(At)),
            ("eps", f"{eps:.3f}" if eps is not None else "—"),
            ("nfz", f"{nfz:.3f}" if nfz is not None else "—"),
            ("F", _fmt_force(F)),
            ("c* efficiency", f"{100.0 * eta_cstar:.2f} %" if eta_cstar is not None else "—"),
            ("Cf efficiency", f"{100.0 * eta_cf:.2f} %" if eta_cf is not None else "—"),
        ]

        cda_rows = [
            ("Fuel throttle CdA", _fmt_area(CdA_f_sys)),
            ("Ox throttle CdA", _fmt_area(CdA_ox_sys)),
            ("Fuel injector CdA", _fmt_area(CdA_f_inj)),
            ("Ox injector CdA", _fmt_area(CdA_ox_inj)),
        ]

        return (
            f"\n================ {self.name} =================\n"
            f"\n[Pressures / Temperatures / Densities]\n"
            f"{_table(state_rows, headers=('Node', 'Fluid', 'Pressure', 'Temperature', 'Density'))}\n"
            f"\n[Mass Flow]\n{_table(flow_rows, headers=('Quantity', 'Value'))}\n"
            f"\n[Nozzle / Performance]\n{_table(geom_rows, headers=('Parameter', 'Value'))}\n"
            f"\n[CdA Summary]\n{_table(cda_rows, headers=('Element', 'Value'))}\n"
            f"================================================\n"
        )

    @classmethod
    def _get_attr_by_path(cls, obj, path: str) -> float:
        """
        Resolve a dotted path like 'FuelInjectorManifold.p' on obj.
        Only supports 2-level paths to match your Balance style.
        """
        parts = path.split(".")
        if len(parts) != 2:
            raise ValueError(f"Expected 'Object.attr' path, got {path!r}")
        comp, attr = parts
        return float(getattr(getattr(obj, comp), attr))


    @classmethod
    def _set_attr_by_path(cls, obj, path: str, value: float) -> None:
        parts = path.split(".")
        if len(parts) != 2:
            raise ValueError(f"Expected 'Object.attr' path, got {path!r}")
        comp, attr = parts
        setattr(getattr(obj, comp), attr, float(value))


    def get_x0(self) -> list[float]:
        """
        Default initial guess for steady_state(), based on STEADY_STATE_X0_PATHS.
        """
        return [self._get_attr_by_path(self, p) for p in self.STEADY_STATE_X0_PATHS]


    def solved_x0(self) -> list[float]:
        """
        Warm-start x0 extracted from a SOLVED TestStand.
        Uses the same STEADY_STATE_X0_PATHS contract.
        """
        return [self._get_attr_by_path(self, p) for p in self.STEADY_STATE_X0_PATHS]
    


    def steady_state(
        self,
        x0: list[float] | None = None,
        tol: float = 1e-9,
        maxfev: int | None = None,
    ) -> "TestStand":
        """
        Solve for steady-state injector-manifold pressures and chamber pressure.

        Solves the nonlinear system for:
        - P_inj_f  : fuel injector upstream (manifold) pressure [Pa]
        - P_inj_ox : oxidizer injector upstream (manifold) pressure [Pa]
        - Pc       : chamber pressure [Pa]

        The residual enforces:
        1) Fuel continuity:
                mdot_sys_fuel(P_tank_f -> P_inj_f) = mdot_inj_fuel(P_inj_f -> Pc)
        2) Ox continuity:
                mdot_sys_ox(P_tank_ox -> P_inj_ox) = mdot_inj_ox(P_inj_ox -> Pc)
        3) Chamber/nozzle continuity:
                mdot_inj_fuel + mdot_inj_ox = mdot_nozzle_choked(Pc, MR, At, eta_cstar)

        After convergence, this method returns a NEW `TestStand` instance (deep-copied)
        with solved pressures and computed mass flows written into the relevant components.
        If ambient pressure is available (via `self.Ambient.p`), it also
        computes nozzle thrust using RocketCEA Cf.

        Parameters
        ----------
        x0 : list[float] or None, optional
            Initial guess [P_inj_f, P_inj_ox, Pc] in Pa.
            If None, uses [FuelInjectorManifold.p, OxInjectorManifold.p, MainChamber.p].
        tol : float, optional
            Root-finder tolerance for SciPy `root(method="hybr")`. Default 1e-9.
        maxfev : int or None, optional
            Maximum function evaluations for the solver. If None, SciPy default.

        Returns
        -------
        TestStand
            A NEW TestStand instance containing solved pressures, mass flow rates,
            mixture ratio, and (optionally) thrust. The original object is not mutated.

        Raises
        ------
        ValueError
            If required physical inputs are non-positive (areas, densities, CdAs, etc.).
        RuntimeError
            If SciPy `root` fails to converge.

        Notes
        -----
        - For continuity residuals, the density used for the CdA equation depends on whether 
          the densities for the injector manifolds are manually set or not. The densities are
          considered to be manually set if the temperature is not the default 300.000001 as given
          in the component constructor. This feature can be used to account for potential
          variations in the propellant densities in the injector manifolds, like due to regen 
          cooling, for example.
        - Feed and injector flows are modeled as incompressible CdA elements.
        - Nozzle mass flow is assumed choked and computed via c* (RocketCEA) and eta_cstar.
        - Ensure Pc/Pamb units match what your RocketCEA wrapper expects.
        """

        # --- Basic validation ---

        if len(self.STEADY_STATE_X0_PATHS) != 3:
            raise ValueError(
                "This steady_state() implementation currently solves exactly 3 unknowns "
                "(FuelInjectorManifold.p, OxInjectorManifold.p, MainChamber.p). "
                "This likely means the TestStand layout has been altered to include more volume nodes. "
                "The number of residuals must now be updated. "
                "Update residual(x) if you change STEADY_STATE_X0_PATHS."
            )
        
        if self.TCA.At <= 0:
            raise ValueError("Nozzle throat area At must be > 0.")
        if self.MainChamber.eta_cstar <= 0:
            raise ValueError("MainChamber.eta_cstar must be > 0.")
        if self.FuelTank.p <= 0 or self.OxTank.p <= 0:
            raise ValueError("Tank pressures must be > 0 Pa.")
        if self.FuelTank.T <= 0 or self.OxTank.T <= 0:
            raise ValueError("Tank temperatures must be > 0 K.")
        if self.FuelThrottleValve.CdA <= 0 or self.OxThrottleValve.CdA <= 0:
            raise ValueError("Throttle valve CdA values must be > 0.")
        if self.FuelInjector.CdA <= 0 or self.OxInjector.CdA <= 0:
            raise ValueError("Injector CdA values must be > 0.")

        fuel_name = self.FuelTank.propellant
        ox_name = self.OxTank.propellant
        cea_obj = self._cea_obj

        P_tank_f = float(self.FuelTank.p)
        P_tank_ox = float(self.OxTank.p)

        '''
        temp_rho_f = float(get_density(fluid_name=fuel_name, 
                                       pressure=self.FuelTank.p, 
                                       temperature=self.FuelTank.T))
        temp_rho_ox = float(get_density(fluid_name=ox_name, 
                                       pressure=self.OxTank.p, 
                                       temperature=self.OxTank.T))

        if self.FuelInjectorManifold.T == 300.000001:
            rho_f1 = temp_rho_f
            rho_f2 = rho_f1
        else:
            rho_f2 = float(get_density(fluid_name=fuel_name, 
                                       pressure=self.FuelInjectorManifold.p, 
                                       temperature=self.FuelInjectorManifold.T))
            rho_f1 = (rho_f2 + temp_rho_f)/2

        
        if self.OxInjectorManifold.T == 300.000001:
            rho_ox1 = temp_rho_ox
            rho_ox2 = rho_ox1
        else:
            rho_ox2 = float(get_density(fluid_name=ox_name, 
                                       pressure=self.OxInjectorManifold.p, 
                                       temperature=self.OxInjectorManifold.T))
            rho_ox1 = (rho_ox2 + temp_rho_ox)/2'''


        sys_CdA_f = float(self.FuelThrottleValve.CdA)
        sys_CdA_ox = float(self.OxThrottleValve.CdA)
        inj_CdA_f = float(self.FuelInjector.CdA)
        inj_CdA_ox = float(self.OxInjector.CdA)

        At = float(self.TCA.At)
        eta_cstar = float(self.MainChamber.eta_cstar)

        def residual(x: np.ndarray) -> np.ndarray:
            P_inj_f, P_inj_ox, Pc = map(float, x)

            # Keep solver away from non-physical space
            if P_inj_f <= 0 or P_inj_ox <= 0 or Pc <= 0:
                return np.array([1e12, 1e12, 1e12], dtype=float)


            try:
                temp_rho_f = float(get_density(
                    fluid_name=fuel_name,
                    pressure=P_tank_f,
                    temperature=self.FuelTank.T
                ))
                temp_rho_ox = float(get_density(
                    fluid_name=ox_name,
                    pressure=P_tank_ox,
                    temperature=self.OxTank.T
                ))

                if self.FuelInjectorManifold.T == 300.000001:
                    rho_f1 = temp_rho_f
                    rho_f2 = rho_f1
                else:
                    rho_f2 = float(get_density(
                        fluid_name=fuel_name,
                        pressure=P_inj_f,
                        temperature=self.FuelInjectorManifold.T
                    ))
                    rho_f1 = 0.5 * (rho_f2 + temp_rho_f)

                if self.OxInjectorManifold.T == 300.000001:
                    rho_ox1 = temp_rho_ox
                    rho_ox2 = rho_ox1
                else:
                    rho_ox2 = float(get_density(
                        fluid_name=ox_name,
                        pressure=P_inj_ox,
                        temperature=self.OxInjectorManifold.T
                    ))
                    rho_ox1 = 0.5 * (rho_ox2 + temp_rho_ox)

            except Exception:
                return np.array([1e12, 1e12, 1e12], dtype=float)


            sys_fuel_mdot = incompressible_CdA_equation(P_tank_f, P_inj_f, rho_f1, sys_CdA_f)
            sys_ox_mdot = incompressible_CdA_equation(P_tank_ox, P_inj_ox, rho_ox1, sys_CdA_ox)

            inj_fuel_mdot = incompressible_CdA_equation(P_inj_f, Pc, rho_f2, inj_CdA_f)
            inj_ox_mdot = incompressible_CdA_equation(P_inj_ox, Pc, rho_ox2, inj_CdA_ox)

            # Guard MR
            # Reject non-physical injector flows before calling CEA
            if inj_fuel_mdot <= 0 or inj_ox_mdot <= 0:
                return np.array([1e12, 1e12, 1e12], dtype=float)

            MR = inj_ox_mdot / inj_fuel_mdot

            # Also keep MR in a sane band for CEA 
            if not (0.5 <= MR <= 6.0):
                return np.array([1e12, 1e12, 1e12], dtype=float)

            # Use the correct function name in your codebase:
            # If your helper is named choked_mass_flow, use that.
            tca_mdot = choked_nozzle_mass_flow(Pc, MR, At, eta_cstar, cea_obj)

            return np.array([
                sys_fuel_mdot - inj_fuel_mdot,
                sys_ox_mdot - inj_ox_mdot,
                (inj_fuel_mdot + inj_ox_mdot) - tca_mdot
            ], dtype=float)

        if x0 is None:
            x0 = self.get_x0()

        expected = len(self.STEADY_STATE_X0_PATHS)
        if len(x0) != expected:
            raise ValueError(
                f"steady_state expected x0 of length {expected} "
                f"(matching STEADY_STATE_X0_PATHS), got {len(x0)}."
            )

        options = {"xtol": tol}
        if maxfev is not None:
            options["maxfev"] = int(maxfev)

        sol = root(residual, x0=np.array(x0, dtype=float), method="hybr", options=options)

        if not sol.success:
            raise RuntimeError(
                "Steady-state solve failed.\n"
                f"message: {sol.message}\n"
                f"x0     : {x0}\n"
                f"x      : {sol.x}"
            )

        P_inj_f_sol, P_inj_ox_sol, Pc_sol = map(float, sol.x)

        solved = copy.deepcopy(self)

        if solved.FuelInjectorManifold.T == 300.000001:
            solved.FuelInjectorManifold.T = solved.FuelTank.T

        if solved.OxInjectorManifold.T == 300.000001:
            solved.OxInjectorManifold.T = solved.OxTank.T

        rho_f = float(get_density(fluid_name=fuel_name, 
                                  pressure=P_inj_f_sol, 
                                  temperature=solved.FuelInjectorManifold.T))

        rho_ox = float(get_density(fluid_name=ox_name, 
                                  pressure=P_inj_ox_sol, 
                                  temperature=solved.OxInjectorManifold.T))

        # Compute final injector mdots at solved pressures
        solved_fuel_mdot = incompressible_CdA_equation(P_inj_f_sol, Pc_sol, rho_f, inj_CdA_f)
        solved_ox_mdot = incompressible_CdA_equation(P_inj_ox_sol, Pc_sol, rho_ox, inj_CdA_ox)

        # Write back mdots
        solved.FuelRunline.mdot = solved_fuel_mdot
        solved.OxRunline.mdot = solved_ox_mdot

        solved.FuelThrottleValve.mdot = solved_fuel_mdot
        solved.OxThrottleValve.mdot = solved_ox_mdot

        # Write back manifold pressures
        solved.FuelInjectorManifold.p = P_inj_f_sol
        solved.OxInjectorManifold.p = P_inj_ox_sol

        # Write back injector mdots
        solved.FuelInjector.mdot = solved_fuel_mdot
        solved.OxInjector.mdot = solved_ox_mdot

        # Chamber state
        solved.MainChamber.p = Pc_sol
        MR_sol = (solved_ox_mdot / solved_fuel_mdot) if solved_fuel_mdot > 0 else float("inf")
        if hasattr(solved.MainChamber, "MR"):
            solved.MainChamber.MR = MR_sol

        solved.FuelInjector.stiffness = (P_inj_f_sol - Pc_sol) / Pc_sol
        solved.OxInjector.stiffness = (P_inj_ox_sol - Pc_sol) / Pc_sol

        # Nozzle state
        solved.TCA.mdot = solved_fuel_mdot + solved_ox_mdot

        solved.TCA.F = choked_nozzle_thrust(
            Pc_sol, MR_sol, solved.TCA.At, self.Ambient.p,
            solved.TCA.eps, solved.TCA.eta_cf, solved.TCA.nfz, cea_obj
        )

        return solved
    


    def steady_state_with_balance(
        self,
        balance: Balance,
        *,
        x0: list[float] | None = None,
        max_iter: int = 60,
        bracket_expand: int = 10,
        fail_penalty: float | None = None,   # kept for compatibility; currently unused
        bracket_samples: int = 8,
        verbose: bool = False
    ) -> "TestStand":
        """
        Solve a 1D "balance" problem by tuning a single scalar knob until a target
        output is met, using repeated steady-state solves and bisection.

        Conceptually, this finds a knob value k such that:

            measure_fn( steady_state(TestStand with knob=k) ) ≈ target

        where the Balance object defines:
        - what attribute to tune (balance.tune_set)
        - what quantity to measure (balance.measure_fn)
        - the target value (balance.target)
        - the allowable knob search bounds (balance.bounds)
        - the convergence tolerance on the measurement error (balance.tol)

        Method Overview
        ---------------
        1) Bracketing:
        Attempts to find two knob values [a, b] within bounds such that the signed
        error changes sign:

            err(k) = measure_fn(solved_ts(k)) - target
            err(a) * err(b) <= 0

        - First tries the endpoints (lo, hi).
        - If that fails, scans `bracket_samples` evenly spaced points in [lo, hi]
            and looks for a sign change between solvable points.
        - If bracketing still fails, optionally expands the bounds outward up to
            `bracket_expand` times (lo *= 0.5, hi *= 2.0) and retries.

        2) Bisection:
        With a valid bracket [a, b], repeatedly bisects:
            mid = (a + b)/2
        evaluates err(mid), and keeps the half-interval that preserves the sign
        change. Stops when |err(mid)| < balance.tol or after `max_iter`.

        Robustness Features
        -------------------
        - Warm-starting:
        Each successful solve updates `last_x0` using `solved.solved_x0()`.
        This makes subsequent steady_state() calls converge faster and more reliably.
        - Handling unsolvable midpoints:
        If steady_state fails at the midpoint, the algorithm nudges the evaluation
        point toward a solvable side (mid->(mid+a)/2, then mid->(mid+b)/2).

        Parameters
        ----------
        balance : Balance
            Defines the knob to tune, measurement to evaluate, target value,
            knob bounds, and tolerance.

        x0 : list[float] | None, keyword-only
            Initial guess passed into steady_state() on the first evaluation.
            If None, uses `self.get_x0()`.

        max_iter : int, keyword-only
            Maximum number of bisection iterations after a bracket is found.

        bracket_expand : int, keyword-only
            Number of times the [lo, hi] bounds may be expanded outward if a valid
            sign-changing bracket cannot be found initially.

        fail_penalty : float | None, keyword-only
            Currently unused (kept for compatibility). If you later decide to treat
            steady_state failures as a large signed error instead of skipping them,
            this value could be used as the magnitude of that penalty.

        bracket_samples : int, keyword-only
            Number of evenly spaced sample points inside [lo, hi] used to find a
            sign-changing bracket when endpoints are not sufficient.

        verbose : bool, keyword-only
            If True, prints diagnostic information when steady_state fails at a knob.

        Returns
        -------
        TestStand
            A NEW solved TestStand instance corresponding to the knob value that
            satisfies the target within tolerance (or best-effort if max_iter reached).

        Raises
        ------
        ValueError
            If the balance bounds are invalid.

        RuntimeError
            If no solvable bracketing points can be found, if the target cannot be
            bracketed (no sign change), or if repeated midpoint evaluations fail.

        Notes
        -----
        - This routine assumes the measured output changes "reasonably" with the knob.
        Bisection requires that the solution be bracketable (i.e., the error crosses 0).
        - The original TestStand object is not mutated; all knob evaluations operate
        on deep-copied versions of `self`.
        """
        lo, hi = map(float, balance.bounds)
        if not (lo > 0 and hi > 0 and hi > lo):
            raise ValueError("balance.bounds must be (lo, hi) with 0 < lo < hi.")

        base = copy.deepcopy(self)
        tol_y = float(balance.tol)

        # Warm-start storage
        last_x0: list[float] | None = copy.deepcopy(x0) if x0 is not None else base.get_x0()

        def try_err_at(knob: float) -> tuple[bool, float, "TestStand" | None]:
            """
            Evaluate the signed error at a given knob value.

            Returns (ok, err, solved_ts):
            ok=False if steady_state fails at this knob.
            """
            nonlocal last_x0
            ts = copy.deepcopy(base)
            balance.tune_set(ts, float(knob))
            try:
                solved = ts.steady_state(x0=last_x0)
                last_x0 = solved.solved_x0()  # warm-start hook
                y = float(balance.measure_fn(solved))
                return True, (y - float(balance.target)), solved
            except Exception as e:
                if verbose:
                    print(f"[solve_with_balance] steady_state failed at knob={knob:.3e}: {e}")
                return False, float("nan"), None

        def find_bracket(lo: float, hi: float) -> tuple[float, float, float, float]:
            """
            Find (a, b, f_a, f_b) with solvable endpoints and f_a * f_b <= 0.
            """
            ok_lo, f_lo, _ = try_err_at(lo)
            ok_hi, f_hi, _ = try_err_at(hi)

            if ok_lo and ok_hi and (f_lo * f_hi <= 0):
                return lo, hi, f_lo, f_hi

            xs = np.linspace(lo, hi, bracket_samples)
            vals: list[tuple[float, float]] = []
            for x in xs:
                ok, fx, _ = try_err_at(float(x))
                if ok and np.isfinite(fx):
                    vals.append((float(x), float(fx)))

            if len(vals) < 2:
                raise RuntimeError(
                    "Could not find ANY solvable points within balance bounds.\n"
                    f"{balance.describe()}\n"
                    "This usually means the bounds are too extreme or x0 is too poor."
                )

            vals.sort(key=lambda t: t[0])
            for (x1, f1), (x2, f2) in zip(vals, vals[1:]):
                if f1 * f2 <= 0:
                    return x1, x2, f1, f2

            fs = [f for _, f in vals]
            raise RuntimeError(
                "Could not bracket target within balance bounds using solvable points.\n"
                f"{balance.describe()}\n"
                f"Solvable sample error range: min={min(fs):.6e}, max={max(fs):.6e}\n"
                "Try widening bounds or choose a different tuning knob."
            )

        expands = 0
        cur_lo, cur_hi = lo, hi
        while True:
            try:
                a, b, f_a, f_b = find_bracket(cur_lo, cur_hi)
                break
            except RuntimeError:
                expands += 1
                if expands > bracket_expand:
                    raise
                cur_lo *= 0.5
                cur_hi *= 2.0

        for _ in range(max_iter):
            mid = 0.5 * (a + b)

            ok_m, f_m, solved_m = try_err_at(mid)
            if not ok_m:
                mid = 0.5 * (mid + a)
                ok_m, f_m, solved_m = try_err_at(mid)
                if not ok_m:
                    mid = 0.5 * (mid + b)
                    ok_m, f_m, solved_m = try_err_at(mid)
                    if not ok_m:
                        raise RuntimeError(
                            "steady_state failed repeatedly near midpoint during balancing.\n"
                            f"{balance.describe()}\n"
                            f"Current bracket: [{a:.3e}, {b:.3e}]"
                        )

            if abs(f_m) < tol_y:
                return solved_m  # type: ignore

            if f_a * f_m <= 0:
                b, f_b = mid, f_m
            else:
                a, f_a = mid, f_m

        ok_m, f_m, solved_m = try_err_at(0.5 * (a + b))
        if ok_m and solved_m is not None:
            return solved_m
        raise RuntimeError(
            "Balance hit max_iter and could not produce a final solved state.\n"
            f"{balance.describe()}\n"
            f"Final bracket: [{a:.3e}, {b:.3e}]"
        )
    

    def __deepcopy__(self, memo):
        """
        Custom deep copy implementation for TestStand.

        This method overrides Python's default deepcopy behavior in order to
        prevent duplication of the cached RocketCEA object used for combustion
        property calculations.

        By default, `copy.deepcopy()` attempts to recursively duplicate every
        attribute of the object. For TestStand instances this would include
        `_cea_obj`, which represents a RocketCEA thermochemistry interface.
        These objects are expensive to construct and may not be safe or useful
        to duplicate.

        Instead, this method performs the following strategy:

        1. Allocate a new uninitialized TestStand instance using `__new__`
        so that the class constructor (`__init__`) is **not executed**.
        This avoids recreating the RocketCEA object.

        2. Recursively deepcopy all normal attributes (tanks, valves,
        manifolds, nozzle, etc.) so that the copied TestStand is
        completely independent of the original.

        3. Copy the `_cea_obj` attribute **by reference**, not by value.
        All TestStand copies therefore share the same cached RocketCEA
        object.

        This behavior ensures that:

        • TestStand copies remain lightweight and fast.
        • RocketCEA initialization occurs only once per propellant pair.
        • Steady-state solves and optimization routines that rely on
        deep copies (such as system design solvers) remain efficient.


        ----------------------------------------------------------------------
        Parameters
        ----------------------------------------------------------------------
        memo : dict
            Internal dictionary used by Python's deepcopy machinery to
            track objects that have already been copied. This prevents
            infinite recursion when copying complex object graphs.


        ----------------------------------------------------------------------
        Returns
        ----------------------------------------------------------------------
        TestStand
            A deep copy of the TestStand instance where all components
            are independent except for the shared `_cea_obj`.
        """
        cls = self.__class__
        result = cls.__new__(cls)     # bypass __init__
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            if k == "_cea_obj":
                setattr(result, k, v)  # keep shared cached object
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result





    def timestep(self, dt: float = 0.001) -> "TestStand":
        """
        Advance the test stand state by one explicit forward-Euler time step.

        This method evaluates all time derivatives using only the CURRENT state
        stored on `self`, then returns a NEW `TestStand` instance with the
        updated state written into the relevant component attributes.

        The original object is not mutated.

        Integration scheme
        ------------------
        This is strict forward Euler:

            x_{n+1} = x_n + f(x_n, t_n) * dt

        which means every derivative is computed only from the step-n state.
        No step-(n+1) quantities are used inside the derivative evaluation.

        What is updated
        ----------------
        The returned TestStand contains updated values for:
        - FuelRunline.mdot
        - OxRunline.mdot
        - FuelThrottleValve.mdot
        - OxThrottleValve.mdot
        - FuelInjectorManifold.p
        - OxInjectorManifold.p
        - FuelInjector.mdot
        - OxInjector.mdot
        - MainChamber.p
        - MainChamber.MR
        - TCA.mdot
        - TCA.F
        - FuelInjector.stiffness
        - OxInjector.stiffness

        Governing model
        ---------------
        The step uses the same lumped-parameter assumptions as the rest of the class:
        - Feed-system momentum dynamics through the runlines:
            d(mdot)/dt = [ΔP - (R/rho) * mdot^2] / L_eq
        - Incompressible injector flow through CdA elements.
        - Injector-manifold density evolution from continuity.
        - Chamber density evolution from total injector inflow minus nozzle outflow.
        - Nozzle mass flow computed from choked flow / c*.
        - Chamber pressure updated from chamber density and mixture ratio.

        Scheduling / commanded inputs
        -----------------------------
        This method intentionally does NOT generate schedules internally.
        Any commanded changes should be applied externally before calling timestep(),
        for example:
            ts.FuelThrottleValve.CdA = ...
            ts.OxThrottleValve.CdA = ...
            ts.FuelInjector.CdA = ...
            ts.MainChamber.eta_cstar = ...

        That keeps this method generic so any attribute can be stepped in time.

        Parameters
        ----------
        dt : float, optional
            Time step in seconds. Must be > 0.

        Returns
        -------
        TestStand
            A new TestStand object containing the advanced state at t + dt.

        Raises
        ------
        ValueError
            If dt <= 0 or if required geometric / physical quantities are non-positive.
        ZeroDivisionError
            If the computed fuel injector flow is zero when forming mixture ratio.
        RuntimeError
            If chamber pressure recovery fails.

        Notes
        -----
        - This method assumes the CURRENT object already contains a physically
        meaningful transient state (pressures, mdots, MR, etc.).
        - Since this is explicit Euler, very small manifold volumes or aggressive
        CdA changes may require a smaller dt for stability.
        - The tank pressures / temperatures are treated as fixed over the step.
        """

        if dt <= 0:
            raise ValueError("dt must be > 0.")

        if self.FuelRunline.A <= 0 or self.OxRunline.A <= 0:
            raise ValueError("Runline cross-sectional areas must be > 0.")

        if self.FuelRunline.L <= 0 or self.OxRunline.L <= 0:
            raise ValueError("Runline lengths must be > 0.")

        if self.FuelThrottleValve.CdA <= 0 or self.OxThrottleValve.CdA <= 0:
            raise ValueError("Throttle valve CdA values must be > 0.")

        if self.FuelInjector.CdA <= 0 or self.OxInjector.CdA <= 0:
            raise ValueError("Injector CdA values must be > 0.")

        if self.FuelInjectorManifold.V <= 0 or self.OxInjectorManifold.V <= 0:
            raise ValueError("Injector manifold volumes must be > 0.")

        if self.MainChamber.V <= 0:
            raise ValueError("Main chamber volume must be > 0.")

        if self.TCA.At <= 0:
            raise ValueError("Nozzle throat area At must be > 0.")

        if self.MainChamber.eta_cstar <= 0:
            raise ValueError("MainChamber.eta_cstar must be > 0.")

        # ------------------------------------------------------------------
        # Read current state (step n)
        # ------------------------------------------------------------------
        fuel_sys_L = self.FuelRunline.L / self.FuelRunline.A
        ox_sys_L = self.OxRunline.L / self.OxRunline.A

        fuel_sys_R = 1.0 / (2.0 * self.FuelThrottleValve.CdA**2)
        ox_sys_R = 1.0 / (2.0 * self.OxThrottleValve.CdA**2)

        fuel_tank_rho = float(get_density(
            self.FuelTank.propellant,
            self.FuelTank.p,
            self.FuelTank.T,
        ))
        ox_tank_rho = float(get_density(
            self.OxTank.propellant,
            self.OxTank.p,
            self.OxTank.T,
        ))

        fuel_inj_rho = float(get_density(
            self.FuelTank.propellant,
            self.FuelInjectorManifold.p,
            self.FuelInjectorManifold.T,
        ))
        ox_inj_rho = float(get_density(
            self.OxTank.propellant,
            self.OxInjectorManifold.p,
            self.OxInjectorManifold.T,
        ))

        fuel_runline_mdot_n = float(self.FuelRunline.mdot)
        ox_runline_mdot_n = float(self.OxRunline.mdot)

        fuel_inj_mdot_n = float(self.FuelInjector.mdot)
        ox_inj_mdot_n = float(self.OxInjector.mdot)

        chamber_p_n = float(self.MainChamber.p)
        chamber_mr_n = float(self.MainChamber.MR)

        # ------------------------------------------------------------------
        # Derivatives evaluated strictly at step n
        # ------------------------------------------------------------------
        fuel_sys_dm_dt = (
            (self.FuelTank.p - self.FuelInjectorManifold.p)
            - (fuel_sys_R / fuel_tank_rho) * fuel_runline_mdot_n**2
        ) / fuel_sys_L

        ox_sys_dm_dt = (
            (self.OxTank.p - self.OxInjectorManifold.p)
            - (ox_sys_R / ox_tank_rho) * ox_runline_mdot_n**2
        ) / ox_sys_L

        fuel_inj_mdot_eval = float(incompressible_CdA_equation(
            self.FuelInjectorManifold.p,
            chamber_p_n,
            fuel_inj_rho,
            self.FuelInjector.CdA,
        ))
        ox_inj_mdot_eval = float(incompressible_CdA_equation(
            self.OxInjectorManifold.p,
            chamber_p_n,
            ox_inj_rho,
            self.OxInjector.CdA,
        ))

        # Forward-only injector model
        fuel_inj_mdot_eval = max(fuel_inj_mdot_eval, 0.0)
        ox_inj_mdot_eval = max(ox_inj_mdot_eval, 0.0)

        if fuel_inj_mdot_eval <= 0.0:
            raise ValueError(
                "Fuel injector mass flow became nonpositive during timestep; "
                "cannot form a valid mixture ratio."
            )

        mr_eval = ox_inj_mdot_eval / fuel_inj_mdot_eval

        if mr_eval <= 0.0:
            raise ValueError(
                f"Computed invalid mixture ratio MR={mr_eval}. "
                "Check manifold pressures, throttle schedules, and dt."
            )

        tca_mdot_eval = float(choked_nozzle_mass_flow(
            chamber_p_n,
            mr_eval,
            self.TCA.At,
            self.MainChamber.eta_cstar,
            self._cea_obj,
        ))

        fuel_inj_drho_dt = (fuel_runline_mdot_n - fuel_inj_mdot_n) / self.FuelInjectorManifold.V
        ox_inj_drho_dt = (ox_runline_mdot_n - ox_inj_mdot_n) / self.OxInjectorManifold.V

        chamber_rho_n = float(self._cea_obj.get_Chamber_Density(chamber_p_n, chamber_mr_n))
        chamber_drho_dt = (
            fuel_inj_mdot_eval + ox_inj_mdot_eval - tca_mdot_eval
        ) / self.MainChamber.V

        # ------------------------------------------------------------------
        # Forward-Euler updates to step n+1
        # ------------------------------------------------------------------
        fuel_runline_mdot_np1 = fuel_runline_mdot_n + fuel_sys_dm_dt * dt
        ox_runline_mdot_np1 = ox_runline_mdot_n + ox_sys_dm_dt * dt

        fuel_inj_rho_np1 = fuel_inj_rho + fuel_inj_drho_dt * dt
        ox_inj_rho_np1 = ox_inj_rho + ox_inj_drho_dt * dt
        chamber_rho_np1 = chamber_rho_n + chamber_drho_dt * dt

        fuel_inj_p_np1 = float(get_pressure(
            self.FuelTank.propellant,
            fuel_inj_rho_np1,
            self.FuelInjectorManifold.T,
        ))
        ox_inj_p_np1 = float(get_pressure(
            self.OxTank.propellant,
            ox_inj_rho_np1,
            self.OxInjectorManifold.T,
        ))

        # Use the evaluated n-state injector flows to define the advanced chamber composition.
        # This stays explicit and avoids using any n+1 injector calculation inside the step.
        mr_np1 = mr_eval

        try:
            chamber_p_np1 = float(get_chamber_pressure(
                chamber_rho_np1,
                mr_np1,
                self._cea_obj,
            ))
        except TypeError:
            # Fallback in case your helper does not accept cea_obj
            chamber_p_np1 = float(get_chamber_pressure(chamber_rho_np1, mr_np1))
        except Exception as e:
            raise RuntimeError(f"Failed to recover chamber pressure from density and MR: {e}") from e

        # Keep injector/nozzle mdots explicit on this step
        fuel_inj_mdot_np1 = fuel_inj_mdot_eval
        ox_inj_mdot_np1 = ox_inj_mdot_eval
        tca_mdot_np1 = tca_mdot_eval

        # ------------------------------------------------------------------
        # Build advanced state object
        # ------------------------------------------------------------------
        next_state = copy.deepcopy(self)

        next_state.FuelRunline.mdot = fuel_runline_mdot_np1
        next_state.OxRunline.mdot = ox_runline_mdot_np1

        next_state.FuelThrottleValve.mdot = fuel_runline_mdot_np1
        next_state.OxThrottleValve.mdot = ox_runline_mdot_np1

        next_state.FuelInjectorManifold.p = fuel_inj_p_np1
        next_state.OxInjectorManifold.p = ox_inj_p_np1

        next_state.FuelInjector.mdot = fuel_inj_mdot_np1
        next_state.OxInjector.mdot = ox_inj_mdot_np1

        next_state.MainChamber.p = chamber_p_np1
        next_state.MainChamber.MR = mr_np1

        next_state.TCA.mdot = tca_mdot_np1

        if chamber_p_np1 > 0:
            next_state.FuelInjector.stiffness = (fuel_inj_p_np1 - chamber_p_np1) / chamber_p_np1
            next_state.OxInjector.stiffness = (ox_inj_p_np1 - chamber_p_np1) / chamber_p_np1

        # Update thrust consistently with advanced chamber pressure and MR
        next_state.TCA.F = choked_nozzle_thrust(
            chamber_p_np1,
            mr_np1,
            next_state.TCA.At,
            next_state.Ambient.p,
            next_state.TCA.eps,
            next_state.TCA.eta_cf,
            next_state.TCA.nfz,
            next_state._cea_obj,
        )

        return next_state
    

    def check_max_throttlable_condition(self, max_fuel_CdA, max_ox_CdA):
        """
        Evaluate steady-state chamber conditions at the maximum fuel and oxidizer
        throttle CdA values.

        This provides a quick estimate of the upper bound of achievable chamber
        pressure (Pc) and the corresponding mixture ratio (MR) at the maximum
        throttle settings.

        Parameters
        ----------
        max_fuel_CdA : float
            Maximum allowable fuel throttle CdA.
        max_ox_CdA : float
            Maximum allowable oxidizer throttle CdA.

        Returns
        -------
        dict
            Dictionary containing:
                - "fuel_CdA": fuel CdA used (max value)
                - "ox_CdA": oxidizer CdA used (max value)
                - "Pc": resulting steady-state chamber pressure
                - "MR": resulting steady-state mixture ratio

        Notes
        -----
        The original throttle CdA values are restored before returning.
        """

        original_fuel_CdA = self.FuelThrottleValve.CdA
        original_ox_CdA = self.OxThrottleValve.CdA

        try:
            self.FuelThrottleValve.CdA = max_fuel_CdA
            self.OxThrottleValve.CdA = max_ox_CdA

            ss = self.steady_state()

            result = {
                "fuel_CdA": max_fuel_CdA,
                "ox_CdA": max_ox_CdA,
                "Pc": ss.MainChamber.p,
                "MR": ss.MainChamber.MR,
            }

        finally:
            self.FuelThrottleValve.CdA = original_fuel_CdA
            self.OxThrottleValve.CdA = original_ox_CdA

        return result
    



    def generate_alpha_map(
        self,
        MR_target: float,
        Pc_min: float,
        Pc_max: float,
        Pc_step: float,
        fuel_CdA_range: tuple[float, float],
        ox_CdA_range: tuple[float, float],
        x0_cdas: tuple[float, float] | None = None,
        x0_state: list[float] | None = None,
        pc_scale: float = 5.0 * PA_PER_PSI,
        mr_scale: float = 0.01,
        residual_tol: float = 1e-6,
        lsq_xtol: float = 1e-10,
        lsq_ftol: float = 1e-10,
        lsq_gtol: float = 1e-10,
        max_nfev: int = 200,
        return_dataframe: bool = True,
        save_parquet: bool = False,
        parquet_filename: str = "pcmr_map.parquet",
        verbose: bool = False
    ) -> dict[str, np.ndarray] | "pd.DataFrame":
        """
        Generate a steady-state lookup table of throttle CdA values that achieve a
        desired chamber pressure Pc and mixture ratio MR.

        This routine solves, at each requested Pc target, for the two unknowns:

            - FuelThrottleValve.CdA
            - OxThrottleValve.CdA

        such that the solved steady-state TestStand satisfies:

            MainChamber.p  = Pc_target
            MainChamber.MR = MR_target

        The resulting map is intended for controller use, where only chamber pressure
        is measured and the desired mixture ratio is enforced by scheduling both
        throttle valves along a precomputed constant-MR operating line.

        Parameters
        ----------
        MR_target : float
            Desired steady-state mixture ratio to hold throughout the map.
        Pc_min : float
            Minimum chamber pressure in the map [Pa].
        Pc_max : float
            Maximum chamber pressure in the map [Pa].
        Pc_step : float
            Chamber pressure step size [Pa]. The generated Pc targets include both
            ends when possible.
        fuel_CdA_range : tuple[float, float]
            Allowed fuel throttle CdA bounds as (min, max) [m^2].
        ox_CdA_range : tuple[float, float]
            Allowed oxidizer throttle CdA bounds as (min, max) [m^2].
        x0_cdas : tuple[float, float] or None, optional
            Initial guess for (fuel throttle CdA, oxidizer throttle CdA) [m^2].
            If None, uses the current TestStand throttle CdA values.
        x0_state : list[float] or None, optional
            Optional initial guess passed into steady_state() for the inner
            manifold/chamber pressure solve. If None, uses self.get_x0().
            After each successful point, the next solve is warm-started from the
            previous solved state.
        pc_scale : float, optional
            Pressure residual scaling [Pa] used inside the least-squares solve.
            Choose a value representative of an acceptable Pc error magnitude.
        mr_scale : float, optional
            Mixture-ratio residual scaling used inside the least-squares solve.
            Choose a value representative of an acceptable MR error magnitude.
        residual_tol : float, optional
            Maximum allowed absolute scaled residual norm for accepting a map point.
            A point is rejected if norm([r_pc_scaled, r_mr_scaled]) > residual_tol.
        lsq_xtol, lsq_ftol, lsq_gtol : float, optional
            SciPy least_squares tolerances.
        max_nfev : int, optional
            Maximum function evaluations for the CdA solve at each map point.
        verbose : bool, optional
            If True, prints per-point solve progress.

        Returns
        -------
        dict[str, np.ndarray]
            Lookup-table dictionary with controller-friendly arrays:

            {
                "MR_target": scalar array of shape (N,),
                "Pc_target": target chamber pressures [Pa],
                "Pc_target_psia": target chamber pressures [psia],
                "Pc_achieved": solved chamber pressures [Pa],
                "Pc_achieved_psia": solved chamber pressures [psia],
                "MR_achieved": solved mixture ratios [-],
                "fuel_CdA": fuel throttle CdA trims [m^2],
                "fuel_CdA_cm2": fuel throttle CdA trims [cm^2],
                "ox_CdA": oxidizer throttle CdA trims [m^2],
                "ox_CdA_cm2": oxidizer throttle CdA trims [cm^2],
                "fuel_mdot": solved fuel mass flow [kg/s],
                "ox_mdot": solved oxidizer mass flow [kg/s],
                "success": boolean success flags
            }

            All arrays are ordered by increasing Pc_target and can be directly used
            with np.interp(...) in the controller.

        Raises
        ------
        ValueError
            If inputs are invalid, bounds are malformed, or no valid Pc grid can be made.
        RuntimeError
            If no map points can be solved successfully.

        Notes
        -----
        - Uses scipy.optimize.least_squares with bounds, which is more robust than
        an unconstrained root solve for physical CdA variables.
        - Uses continuation / warm-starting:
            * the previous successful CdA pair seeds the next CdA solve
            * the previous successful steady-state x0 seeds the next steady_state() solve
        - The original TestStand object is not mutated.
        - The returned map is intended to serve as a nominal constant-MR throttle line.
        A pressure controller can then interpolate these trims and apply a small
        common-mode correction around them.
        """

        MR_target = float(MR_target)
        Pc_min = float(Pc_min)
        Pc_max = float(Pc_max)
        Pc_step = float(Pc_step)
        pc_scale = float(pc_scale)
        mr_scale = float(mr_scale)
        residual_tol = float(residual_tol)

        fuel_CdA_min, fuel_CdA_max = map(float, fuel_CdA_range)
        ox_CdA_min, ox_CdA_max = map(float, ox_CdA_range)

        if MR_target <= 0.0:
            raise ValueError(f"MR_target must be > 0. Got {MR_target}.")
        if Pc_min <= 0.0 or Pc_max <= 0.0:
            raise ValueError(f"Pc_min and Pc_max must be > 0. Got {Pc_min}, {Pc_max}.")
        if Pc_max < Pc_min:
            raise ValueError(f"Pc_max must be >= Pc_min. Got Pc_min={Pc_min}, Pc_max={Pc_max}.")
        if Pc_step <= 0.0:
            raise ValueError(f"Pc_step must be > 0. Got {Pc_step}.")
        if pc_scale <= 0.0:
            raise ValueError(f"pc_scale must be > 0. Got {pc_scale}.")
        if mr_scale <= 0.0:
            raise ValueError(f"mr_scale must be > 0. Got {mr_scale}.")
        if fuel_CdA_min <= 0.0 or fuel_CdA_max <= 0.0 or fuel_CdA_max <= fuel_CdA_min:
            raise ValueError(
                "fuel_CdA_range must be (min, max) with 0 < min < max. "
                f"Got {fuel_CdA_range}."
            )
        if ox_CdA_min <= 0.0 or ox_CdA_max <= 0.0 or ox_CdA_max <= ox_CdA_min:
            raise ValueError(
                "ox_CdA_range must be (min, max) with 0 < min < max. "
                f"Got {ox_CdA_range}."
            )

        # Build inclusive Pc target grid.
        n_steps = int(np.floor((Pc_max - Pc_min) / Pc_step + 1e-12))
        Pc_targets = Pc_min + np.arange(n_steps + 1, dtype=float) * Pc_step
        if Pc_targets.size == 0:
            raise ValueError("No Pc targets generated. Check Pc_min, Pc_max, and Pc_step.")
        if Pc_targets[-1] < Pc_max - 1e-12:
            Pc_targets = np.append(Pc_targets, Pc_max)

        base = copy.deepcopy(self)

        if x0_cdas is None:
            x_cda_guess = np.array(
                [
                    float(base.FuelThrottleValve.CdA),
                    float(base.OxThrottleValve.CdA),
                ],
                dtype=float,
            )
        else:
            x_cda_guess = np.array([float(x0_cdas[0]), float(x0_cdas[1])], dtype=float)

        if not (fuel_CdA_min <= x_cda_guess[0] <= fuel_CdA_max):
            raise ValueError(
                f"Initial fuel CdA guess {x_cda_guess[0]:.6e} is outside bounds "
                f"[{fuel_CdA_min:.6e}, {fuel_CdA_max:.6e}]."
            )
        if not (ox_CdA_min <= x_cda_guess[1] <= ox_CdA_max):
            raise ValueError(
                f"Initial ox CdA guess {x_cda_guess[1]:.6e} is outside bounds "
                f"[{ox_CdA_min:.6e}, {ox_CdA_max:.6e}]."
            )

        last_state_x0 = copy.deepcopy(x0_state) if x0_state is not None else base.get_x0()

        rows: list[dict[str, float | bool]] = []

        bounds_lo = np.array([fuel_CdA_min, ox_CdA_min], dtype=float)
        bounds_hi = np.array([fuel_CdA_max, ox_CdA_max], dtype=float)

        for Pc_target in Pc_targets:
            Pc_target = float(Pc_target)

            def residuals(x: np.ndarray) -> np.ndarray:
                fuel_cda = float(x[0])
                ox_cda = float(x[1])

                ts_try = copy.deepcopy(base)
                ts_try.FuelThrottleValve.CdA = fuel_cda
                ts_try.OxThrottleValve.CdA = ox_cda

                try:
                    solved = ts_try.steady_state(x0=last_state_x0)
                except Exception:
                    return np.array([1.0e6, 1.0e6], dtype=float)

                r_pc = (float(solved.MainChamber.p) - Pc_target) / pc_scale
                r_mr = (float(solved.MainChamber.MR) - MR_target) / mr_scale
                return np.array([r_pc, r_mr], dtype=float)

            lsq = least_squares(
                residuals,
                x0=x_cda_guess,
                bounds=(bounds_lo, bounds_hi),
                xtol=lsq_xtol,
                ftol=lsq_ftol,
                gtol=lsq_gtol,
                max_nfev=max_nfev,
            )

            point_success = False
            fuel_cda_sol = float(lsq.x[0])
            ox_cda_sol = float(lsq.x[1])

            Pc_achieved = np.nan
            MR_achieved = np.nan
            fuel_mdot = np.nan
            ox_mdot = np.nan

            try:
                ts_sol = copy.deepcopy(base)
                ts_sol.FuelThrottleValve.CdA = fuel_cda_sol
                ts_sol.OxThrottleValve.CdA = ox_cda_sol
                ts_sol = ts_sol.steady_state(x0=last_state_x0)

                r_pc_final = (float(ts_sol.MainChamber.p) - Pc_target) / pc_scale
                r_mr_final = (float(ts_sol.MainChamber.MR) - MR_target) / mr_scale
                residual_norm = float(np.linalg.norm([r_pc_final, r_mr_final]))

                if lsq.success and np.isfinite(residual_norm) and residual_norm <= residual_tol:
                    point_success = True
                    Pc_achieved = float(ts_sol.MainChamber.p)
                    MR_achieved = float(ts_sol.MainChamber.MR)
                    fuel_mdot = float(ts_sol.FuelInjector.mdot)
                    ox_mdot = float(ts_sol.OxInjector.mdot)

                    # Continuation / warm start for next point.
                    x_cda_guess = np.array([fuel_cda_sol, ox_cda_sol], dtype=float)
                    last_state_x0 = ts_sol.solved_x0()

                    if verbose:
                        print(
                            "[generate_PcMR_map] OK  "
                            f"Pc_target={Pc_target / PA_PER_PSI:8.3f} psia, "
                            f"Pc={Pc_achieved / PA_PER_PSI:8.3f} psia, "
                            f"MR={MR_achieved:7.5f}, "
                            f"fuel_CdA={fuel_cda_sol * 1e4:8.5f} cm^2, "
                            f"ox_CdA={ox_cda_sol * 1e4:8.5f} cm^2"
                        )
                else:
                    if verbose:
                        print(
                            "[generate_PcMR_map] FAIL "
                            f"Pc_target={Pc_target / PA_PER_PSI:8.3f} psia, "
                            f"lsq.success={lsq.success}, "
                            f"residual_norm={residual_norm:.3e}"
                        )

            except Exception as e:
                if verbose:
                    print(
                        "[generate_PcMR_map] FAIL "
                        f"Pc_target={Pc_target / PA_PER_PSI:8.3f} psia, "
                        f"exception={e}"
                    )

            rows.append(
                {
                    "MR_target": MR_target,
                    "Pc_target": Pc_target,
                    "Pc_achieved": Pc_achieved,
                    "MR_achieved": MR_achieved,
                    "fuel_CdA": fuel_cda_sol,
                    "ox_CdA": ox_cda_sol,
                    "fuel_mdot": fuel_mdot,
                    "ox_mdot": ox_mdot,
                    "success": point_success,
                }
            )

        success_rows = [row for row in rows if bool(row["success"])]
        if len(success_rows) == 0:
            raise RuntimeError(
                "generate_PcMR_map() could not solve any valid map points. "
                "Try a better initial CdA guess, a narrower Pc range, or looser bounds."
            )

        # Only keep successful points in the final lookup table.
        Pc_target_arr = np.array([float(r["Pc_target"]) for r in success_rows], dtype=float)
        Pc_achieved_arr = np.array([float(r["Pc_achieved"]) for r in success_rows], dtype=float)
        MR_target_arr = np.array([float(r["MR_target"]) for r in success_rows], dtype=float)
        MR_achieved_arr = np.array([float(r["MR_achieved"]) for r in success_rows], dtype=float)
        fuel_CdA_arr = np.array([float(r["fuel_CdA"]) for r in success_rows], dtype=float)
        ox_CdA_arr = np.array([float(r["ox_CdA"]) for r in success_rows], dtype=float)
        fuel_mdot_arr = np.array([float(r["fuel_mdot"]) for r in success_rows], dtype=float)
        ox_mdot_arr = np.array([float(r["ox_mdot"]) for r in success_rows], dtype=float)
        success_arr = np.array([bool(r["success"]) for r in success_rows], dtype=bool)

        df = pd.DataFrame({
            "Pc_target": Pc_target_arr,
            "Pc": Pc_achieved_arr,
            "MR": MR_achieved_arr,
            "fuel_CdA": fuel_CdA_arr,
            "ox_CdA": ox_CdA_arr,
            "mdot_f": fuel_mdot_arr,
            "mdot_ox": ox_mdot_arr,
            "success": success_arr,
        })

        # --------------------------------------------
        # Compute alpha from total mass flow
        # --------------------------------------------
        df["mdot_total"] = df["mdot_f"] + df["mdot_ox"]

        mdot_min = df["mdot_total"].min()
        mdot_max = df["mdot_total"].max()

        if mdot_max <= mdot_min:
            raise RuntimeError("Invalid mdot range for alpha computation.")

        df["alpha"] = (df["mdot_total"] - mdot_min) / (mdot_max - mdot_min)

        # --------------------------------------------
        # Sort by alpha (this defines your manifold)
        # --------------------------------------------
        cols = ["alpha"] + [c for c in df.columns if c != "alpha"]
        df = df[cols]
        df = df.sort_values("alpha").reset_index(drop=True)


        # Optional: Save to parquet with auto-increment
        if save_parquet:
            import os

            base_name, ext = os.path.splitext(parquet_filename)
            if ext == "":
                ext = ".parquet"

            final_name = base_name + ext
            counter = 1

            while os.path.exists(final_name):
                final_name = f"{base_name}_{counter}{ext}"
                counter += 1

            df.to_parquet(final_name, index=False)

            if verbose:
                print(f"[generate_PcMR_map] Saved parquet: {final_name}")


        if return_dataframe:
            return df
        else:
            return df.to_dict(orient="list")
        


    def generate_ox_cda_map(
        self,
        fuel_CdA: float,
        ox_CdA_range: tuple[float, float],
        ox_CdA_step: float,
        x0_state: list[float] | None = None,
        return_dataframe: bool = True,
        save_parquet: bool = False,
        parquet_filename: str = "ox_cda_map.parquet",
        verbose: bool = False,
    ) -> dict[str, np.ndarray] | "pd.DataFrame":
        """
        Generate a steady-state lookup table of oxidizer throttle CdA vs chamber pressure
        for a fixed fuel throttle CdA.

        This routine:
            - holds FuelThrottleValve.CdA constant
            - sweeps OxThrottleValve.CdA over a specified range
            - solves steady_state() at each ox CdA
            - records the resulting chamber pressure Pc and mixture ratio MR

        All exported values are in SI units:
            - CdA in m^2
            - Pc in Pa
            - mdot in kg/s

        Parameters
        ----------
        fuel_CdA : float
            Fixed fuel throttle CdA [m^2].
        ox_CdA_range : tuple[float, float]
            Sweep bounds for oxidizer throttle CdA as (min, max) [m^2].
        ox_CdA_step : float
            Step size for oxidizer throttle CdA sweep [m^2].
        x0_state : list[float] or None, optional
            Optional initial guess passed into steady_state(). After each successful
            point, the next solve is warm-started using solved_x0().
        return_dataframe : bool, optional
            If True, return a pandas DataFrame. Otherwise return a dict of lists.
        save_parquet : bool, optional
            If True, save the resulting DataFrame to parquet.
        parquet_filename : str, optional
            Base parquet filename.
        verbose : bool, optional
            If True, print per-point progress.

        Returns
        -------
        pandas.DataFrame or dict[str, np.ndarray]
            Columns:
                - fuel_CdA   [m^2]
                - ox_CdA     [m^2]
                - Pc         [Pa]
                - MR         [-]
                - mdot_f     [kg/s]
                - mdot_ox    [kg/s]
                - mdot_total [kg/s]
                - success    [bool]

        Raises
        ------
        ValueError
            If inputs are invalid.
        RuntimeError
            If no valid points are solved.
        """

        fuel_CdA = float(fuel_CdA)
        ox_CdA_min, ox_CdA_max = map(float, ox_CdA_range)
        ox_CdA_step = float(ox_CdA_step)

        if fuel_CdA <= 0.0:
            raise ValueError(f"fuel_CdA must be > 0. Got {fuel_CdA}.")
        if ox_CdA_min <= 0.0 or ox_CdA_max <= 0.0 or ox_CdA_max <= ox_CdA_min:
            raise ValueError(
                "ox_CdA_range must be (min, max) with 0 < min < max. "
                f"Got {ox_CdA_range}."
            )
        if ox_CdA_step <= 0.0:
            raise ValueError(f"ox_CdA_step must be > 0. Got {ox_CdA_step}.")

        # Inclusive ox CdA grid
        n_steps = int(np.floor((ox_CdA_max - ox_CdA_min) / ox_CdA_step + 1e-12))
        ox_CdA_targets = ox_CdA_min + np.arange(n_steps + 1, dtype=float) * ox_CdA_step
        if ox_CdA_targets.size == 0:
            raise ValueError("No ox_CdA targets generated. Check ox_CdA_range and ox_CdA_step.")
        if ox_CdA_targets[-1] < ox_CdA_max - 1e-12:
            ox_CdA_targets = np.append(ox_CdA_targets, ox_CdA_max)

        base = copy.deepcopy(self)
        base.FuelThrottleValve.CdA = fuel_CdA

        last_state_x0 = copy.deepcopy(x0_state) if x0_state is not None else base.get_x0()

        rows: list[dict[str, float | bool]] = []

        for ox_cda in ox_CdA_targets:
            ox_cda = float(ox_cda)

            point_success = False
            Pc_achieved = np.nan
            MR_achieved = np.nan
            fuel_mdot = np.nan
            ox_mdot = np.nan

            try:
                ts_sol = copy.deepcopy(base)
                ts_sol.OxThrottleValve.CdA = ox_cda
                ts_sol = ts_sol.steady_state(x0=last_state_x0)

                Pc_achieved = float(ts_sol.MainChamber.p)
                MR_achieved = float(ts_sol.MainChamber.MR)
                fuel_mdot = float(ts_sol.FuelInjector.mdot)
                ox_mdot = float(ts_sol.OxInjector.mdot)

                point_success = True

                # Warm start next solve
                last_state_x0 = ts_sol.solved_x0()

            except Exception:
                point_success = False

            if verbose:
                print(
                    "[generate_ox_cda_map] "
                    f"ox_CdA={ox_cda:.6e} m^2, "
                    f"Pc={Pc_achieved:.6e} Pa, "
                    f"MR={MR_achieved:.6f}, "
                    f"success={point_success}"
                )

            rows.append(
                {
                    "fuel_CdA": fuel_CdA,                          # m^2
                    "ox_CdA": ox_cda,                              # m^2
                    "Pc": Pc_achieved,                             # Pa
                    "MR": MR_achieved,                             # -
                    "mdot_f": fuel_mdot,                           # kg/s
                    "mdot_ox": ox_mdot,                            # kg/s
                    "mdot_total": (
                        fuel_mdot + ox_mdot
                        if np.isfinite(fuel_mdot) and np.isfinite(ox_mdot)
                        else np.nan
                    ),
                    "success": point_success,
                }
            )

        success_rows = [row for row in rows if bool(row["success"])]
        if len(success_rows) == 0:
            raise RuntimeError(
                "generate_ox_cda_map() could not produce any valid solved points."
            )

        df = pd.DataFrame({
            "fuel_CdA": np.array([float(r["fuel_CdA"]) for r in success_rows], dtype=float),
            "ox_CdA": np.array([float(r["ox_CdA"]) for r in success_rows], dtype=float),
            "Pc": np.array([float(r["Pc"]) for r in success_rows], dtype=float),
            "MR": np.array([float(r["MR"]) for r in success_rows], dtype=float),
            "mdot_f": np.array([float(r["mdot_f"]) for r in success_rows], dtype=float),
            "mdot_ox": np.array([float(r["mdot_ox"]) for r in success_rows], dtype=float),
            "mdot_total": np.array([float(r["mdot_total"]) for r in success_rows], dtype=float),
            "success": np.array([bool(r["success"]) for r in success_rows], dtype=bool),
        })

        # Sort by increasing chamber pressure for lookup use
        df = df.sort_values(by="Pc", ascending=True, kind="mergesort").reset_index(drop=True)

        if save_parquet:
            import os

            base_name, ext = os.path.splitext(parquet_filename)
            if ext == "":
                ext = ".parquet"

            final_name = base_name + ext
            counter = 1

            while os.path.exists(final_name):
                final_name = f"{base_name}_{counter}{ext}"
                counter += 1

            # Saved dataframe is SI-only
            df.to_parquet(final_name, index=False)

            if verbose:
                print(f"[generate_ox_cda_map] Saved parquet: {final_name}")

        if return_dataframe:
            return df
        else:
            return df.to_dict(orient="list")
        


    def generate_fuel_cda_map(
        self,
        ox_CdA: float,
        fuel_CdA_range: tuple[float, float],
        fuel_CdA_step: float,
        x0_state: list[float] | None = None,
        return_dataframe: bool = True,
        save_parquet: bool = False,
        parquet_filename: str = "fuel_cda_map.parquet",
        verbose: bool = False,
    ) -> dict[str, np.ndarray] | "pd.DataFrame":
        """
        Generate a steady-state lookup table of fuel throttle CdA vs chamber pressure
        for a fixed oxidizer throttle CdA.

        This routine:
            - holds OxThrottleValve.CdA constant
            - sweeps FuelThrottleValve.CdA over a specified range
            - solves steady_state() at each fuel CdA
            - records the resulting chamber pressure Pc and mixture ratio MR

        All exported values are in SI units:
            - CdA in m^2
            - Pc in Pa
            - mdot in kg/s

        Parameters
        ----------
        ox_CdA : float
            Fixed oxidizer throttle CdA [m^2].
        fuel_CdA_range : tuple[float, float]
            Sweep bounds for fuel throttle CdA as (min, max) [m^2].
        fuel_CdA_step : float
            Step size for fuel throttle CdA sweep [m^2].
        x0_state : list[float] or None, optional
            Optional initial guess passed into steady_state(). After each successful
            point, the next solve is warm-started using solved_x0().
        return_dataframe : bool, optional
            If True, return a pandas DataFrame. Otherwise return a dict of lists.
        save_parquet : bool, optional
            If True, save the resulting DataFrame to parquet.
        parquet_filename : str, optional
            Base parquet filename.
        verbose : bool, optional
            If True, print per-point progress.

        Returns
        -------
        pandas.DataFrame or dict[str, np.ndarray]
            Columns:
                - ox_CdA      [m^2]
                - fuel_CdA    [m^2]
                - Pc          [Pa]
                - MR          [-]
                - mdot_f      [kg/s]
                - mdot_ox     [kg/s]
                - mdot_total  [kg/s]
                - success     [bool]

        Raises
        ------
        ValueError
            If inputs are invalid.
        RuntimeError
            If no valid points are solved.
        """

        ox_CdA = float(ox_CdA)
        fuel_CdA_min, fuel_CdA_max = map(float, fuel_CdA_range)
        fuel_CdA_step = float(fuel_CdA_step)

        if ox_CdA <= 0.0:
            raise ValueError(f"ox_CdA must be > 0. Got {ox_CdA}.")
        if fuel_CdA_min <= 0.0 or fuel_CdA_max <= 0.0 or fuel_CdA_max <= fuel_CdA_min:
            raise ValueError(
                "fuel_CdA_range must be (min, max) with 0 < min < max. "
                f"Got {fuel_CdA_range}."
            )
        if fuel_CdA_step <= 0.0:
            raise ValueError(f"fuel_CdA_step must be > 0. Got {fuel_CdA_step}.")

        # Inclusive fuel CdA grid
        n_steps = int(np.floor((fuel_CdA_max - fuel_CdA_min) / fuel_CdA_step + 1e-12))
        fuel_CdA_targets = fuel_CdA_min + np.arange(n_steps + 1, dtype=float) * fuel_CdA_step
        if fuel_CdA_targets.size == 0:
            raise ValueError("No fuel_CdA targets generated. Check fuel_CdA_range and fuel_CdA_step.")
        if fuel_CdA_targets[-1] < fuel_CdA_max - 1e-12:
            fuel_CdA_targets = np.append(fuel_CdA_targets, fuel_CdA_max)

        base = copy.deepcopy(self)
        base.OxThrottleValve.CdA = ox_CdA

        last_state_x0 = copy.deepcopy(x0_state) if x0_state is not None else base.get_x0()

        rows: list[dict[str, float | bool]] = []

        for fuel_cda in fuel_CdA_targets:
            fuel_cda = float(fuel_cda)

            point_success = False
            Pc_achieved = np.nan
            MR_achieved = np.nan
            fuel_mdot = np.nan
            ox_mdot = np.nan

            try:
                ts_sol = copy.deepcopy(base)
                ts_sol.FuelThrottleValve.CdA = fuel_cda
                ts_sol = ts_sol.steady_state(x0=last_state_x0)

                Pc_achieved = float(ts_sol.MainChamber.p)
                MR_achieved = float(ts_sol.MainChamber.MR)
                fuel_mdot = float(ts_sol.FuelInjector.mdot)
                ox_mdot = float(ts_sol.OxInjector.mdot)

                point_success = True

                # Warm start next solve
                last_state_x0 = ts_sol.solved_x0()

            except Exception:
                point_success = False

            if verbose:
                print(
                    "[generate_fuel_cda_map] "
                    f"fuel_CdA={fuel_cda:.6e} m^2, "
                    f"Pc={Pc_achieved:.6e} Pa, "
                    f"MR={MR_achieved:.6f}, "
                    f"success={point_success}"
                )

            rows.append(
                {
                    "ox_CdA": ox_CdA,                                # m^2
                    "fuel_CdA": fuel_cda,                            # m^2
                    "Pc": Pc_achieved,                               # Pa
                    "MR": MR_achieved,                               # -
                    "mdot_f": fuel_mdot,                             # kg/s
                    "mdot_ox": ox_mdot,                              # kg/s
                    "mdot_total": (
                        fuel_mdot + ox_mdot
                        if np.isfinite(fuel_mdot) and np.isfinite(ox_mdot)
                        else np.nan
                    ),
                    "success": point_success,
                }
            )

        success_rows = [row for row in rows if bool(row["success"])]
        if len(success_rows) == 0:
            raise RuntimeError(
                "generate_fuel_cda_map() could not produce any valid solved points."
            )

        df = pd.DataFrame({
            "ox_CdA": np.array([float(r["ox_CdA"]) for r in success_rows], dtype=float),
            "fuel_CdA": np.array([float(r["fuel_CdA"]) for r in success_rows], dtype=float),
            "Pc": np.array([float(r["Pc"]) for r in success_rows], dtype=float),
            "MR": np.array([float(r["MR"]) for r in success_rows], dtype=float),
            "mdot_f": np.array([float(r["mdot_f"]) for r in success_rows], dtype=float),
            "mdot_ox": np.array([float(r["mdot_ox"]) for r in success_rows], dtype=float),
            "mdot_total": np.array([float(r["mdot_total"]) for r in success_rows], dtype=float),
            "success": np.array([bool(r["success"]) for r in success_rows], dtype=bool),
        })

        # Sort by increasing chamber pressure for lookup use
        df = df.sort_values(by="Pc", ascending=True, kind="mergesort").reset_index(drop=True)

        if save_parquet:
            import os

            base_name, ext = os.path.splitext(parquet_filename)
            if ext == "":
                ext = ".parquet"

            final_name = base_name + ext
            counter = 1

            while os.path.exists(final_name):
                final_name = f"{base_name}_{counter}{ext}"
                counter += 1

            # Saved dataframe is SI-only
            df.to_parquet(final_name, index=False)

            if verbose:
                print(f"[generate_fuel_cda_map] Saved parquet: {final_name}")

        if return_dataframe:
            return df
        else:
            return df.to_dict(orient="list")