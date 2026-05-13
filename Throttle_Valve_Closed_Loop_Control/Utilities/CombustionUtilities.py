from rocketcea.cea_obj_w_units import CEA_Obj
from scipy.optimize import root_scalar
from .PROPELLANT_NAMES import PROPELLANT_NAME_BANK
import numpy as np


def normalize_propellant_name(name: str) -> str:
    """
    Convert arbitrary user input into a RocketCEA-compatible propellant name.

    Parameters
    ----------
    name : str
        Input propellant name (case-insensitive, flexible aliases)

    Returns
    -------
    str
        RocketCEA-compatible propellant name

    Raises
    ------
    ValueError
        If the propellant is not recognized
    """
    if not isinstance(name, str):
        raise TypeError("Propellant name must be a string")

    key = name.strip().lower()

    if key in PROPELLANT_NAME_BANK:
        return PROPELLANT_NAME_BANK[key]

    raise ValueError(
        f"Unknown propellant '{name}'. Add it to PROPELLANT_NAME_BANK."
    )



def create_CEA_object(fuel: str = 'RP-1', oxidizer: str = 'LOX') -> CEA_Obj:
    """
    Create and return a RocketCEA `CEA_Obj` configured with SI-based units.

    This helper function initializes a `CEA_Obj` with consistent unit settings
    for propulsion analysis workflows. It is intended for use in chamber,
    injector, and nozzle performance calculations (e.g., c*, Isp, gamma).

    The returned object uses:
        - Temperature: Kelvin
        - c*: m/s
        - Sonic velocity: m/s
        - Enthalpy: J/kg
        - Density: kg/m^3
        - Specific heat: kJ/kg-K
        - Pressure: Pa

    Notes
    -----
    RocketCEA internally expects chamber pressure in psia by default.
    Ensure proper unit conversion when passing pressure values.

    FAC (Finite Area Combustor) support may be added in future versions.

    Parameters
    ----------
    fuel : str, optional
        Fuel propellant name (must match a valid RocketCEA fuel identifier).
        Default is 'LOX'.
    oxidizer : str, optional
        Oxidizer propellant name (must match a valid RocketCEA oxidizer identifier).
        Default is 'RP-1'.

    Returns
    -------
    CEA_Obj
        Configured RocketCEA object ready for thermochemical property evaluation.

    Raises
    ------
    ValueError
        If the specified fuel or oxidizer name is not recognized by RocketCEA.

    Examples
    --------
    >>> cea = create_CEA_object('RP-1', 'LOX')
    >>> cstar = cea.get_Cstar(Pc, MR)
    """
    obj = CEA_Obj(
        oxName=normalize_propellant_name(oxidizer),
        fuelName=normalize_propellant_name(fuel),
        temperature_units='degK',
        cstar_units='m/sec',
        specific_heat_units='kJ/kg degK',
        sonic_velocity_units='m/s',
        enthalpy_units='J/kg',
        density_units='kg/m^3',
        pressure_units='Pa'
    )

    return obj


default_cea_obj = create_CEA_object()



def choked_nozzle_mass_flow(Pc : float, MR: float, At: float, eta_cstar: float = 1.0, cea_obj:CEA_Obj = default_cea_obj) -> float:
    """
    Compute the total choked mass flow rate through a rocket nozzle throat.

    This function assumes ideal choked flow at the throat and uses the
    characteristic velocity (c*) relation:

        mdot = Pc * At / (η_c* * c*_ideal)

    where:
        Pc         = chamber pressure
        At         = throat area
        c*_ideal   = ideal characteristic velocity from CEA
        η_c*     = c* efficiency (accounts for combustion losses)

    Parameters
    ----------
    Pc : float
        Chamber pressure [Pa]. Must be > 0.
    MR : float
        Mixture ratio (O/F). Must be > 0.
    At : float
        Nozzle throat area [m^2]. Must be > 0.
    eta_c* : float, optional
        Characteristic velocity efficiency (dimensionless).
        Default is 1.0 (ideal case).
    cea_obj : CEA_Obj, optional
        RocketCEA object used to compute ideal c*.
        Default is `default_cea_obj`.

    Returns
    -------
    mdot : float
        Total choked mass flow rate [kg/s].

    Notes
    -----
    - Assumes steady, one-dimensional, choked flow at the throat.
    - Losses such as divergence efficiency and discharge coefficient
      are not included unless incorporated into `cstar_eff`.
    - Units must be consistent (SI recommended).
    """
    cstar_ideal = cea_obj.get_Cstar(Pc, MR)
    if eta_cstar <= 0:
        raise ValueError("eta_cstar must be > 0")
    if At <= 0:
        raise ValueError("At must be > 0")
    if Pc <= 0:
        raise ValueError("Pc must be > 0")

    # Guard bad CEA returns
    if (cstar_ideal is None) or (not np.isfinite(cstar_ideal)) or (cstar_ideal <= 0):
        raise ValueError(
            f"Invalid cstar_ideal from CEA: {cstar_ideal} "
            f"(Pc={Pc}, MR={MR})"
        )

    return Pc * At / (eta_cstar * cstar_ideal)


def choked_nozzle_thrust(
    Pc: float,
    MR: float,
    At: float,
    Pamb: float,
    eps: float,
    eta_cf: float = 1.0,
    nfz: int = 0,
    cea_obj: CEA_Obj = default_cea_obj
) -> float:
    """
    Compute thrust for a choked nozzle using RocketCEA thrust coefficient (Cf).

    This function assumes the nozzle is choked and computes thrust via:

        F = η_cf * Cf * Pc * At

    where:
      - Pc is chamber pressure,
      - At is throat area,
      - Cf is the (dimensionless) thrust coefficient from RocketCEA,
      - η_cf is an optional correction/efficiency applied to Cf to account for
        non-ideal nozzle effects (e.g., divergence losses, boundary layer losses,
        nozzle erosion/roughness, off-design effects).

    RocketCEA provides Cf as a function of chamber pressure, mixture ratio, area
    ratio (eps = Ae/At), and ambient pressure, and can compute either equilibrium
    (default) or frozen chemistry cases.

    Parameters
    ----------
    Pc : float
        Chamber pressure [Pa]. Must be > 0.
    MR : float
        Mixture ratio. Must be > 0.
    At : float
        Nozzle throat area [m^2]. Must be > 0.
    Pamb : float
        Ambient (back) pressure [Pa]. Must be >= 0.
        Used by RocketCEA to compute pressure thrust effects through Cf.
    eps : float
        Nozzle expansion ratio Ae/At (dimensionless). Must be > 0.
    eta_cf : float, optional
        Multiplicative efficiency factor applied to Cf (dimensionless).
        Use 1.0 for ideal RocketCEA Cf. Typical realistic values might be ~0.95-0.99
        depending on nozzle design/model fidelity. Default is 1.0.
    nfz : int, optional
        Chemistry model selector:
          - 0 : equilibrium chemistry (RocketCEA `get_PambCf`)
          - 1 : frozen chemistry, fully frozen (RocketCEA `getFrozen_PambCf(..., frozenAtThroat=0)`)
          - 2 : frozen chemistry, frozen from throat (RocketCEA `getFrozen_PambCf(..., frozenAtThroat=1)`)
        Default is 0 (equilibrium).
    cea_obj : CEA_Obj, optional
        RocketCEA object used to compute Cf. Default is `default_cea_obj`.

    Returns
    -------
    thrust : float
        Nozzle thrust [N].

    Raises
    ------
    ValueError
        If any required inputs are non-physical (e.g., Pc <= 0, At <= 0, MR <= 0, eps <= 0,
        eta_cf <= 0, or Pamb < 0).
    ValueError
        If `nfz` is not one of {0, 1, 2}.

    Notes
    -----
    - This function computes thrust using Cf from RocketCEA; it does not directly
      compute exit pressure, exit velocity, or Isp. Those can be derived separately.
    - The formula assumes choked flow at the throat; if the nozzle is unchoked,
      Cf-based thrust computed this way may not be valid.

    Examples
    --------
    >>> F = choked_nozzle_thrust(Pc=3.0e6, MR=2.6, At=1.2e-4, Pamb=101325.0, eps=8.0)
    >>> print(F)
    """
    # --- Validation ---
    if Pc <= 0:
        raise ValueError("Pc must be > 0 (Pa).")
    if MR <= 0:
        raise ValueError("MR must be > 0 (O/F).")
    if At <= 0:
        raise ValueError("At must be > 0 (m^2).")
    if eps <= 0:
        raise ValueError("eps (Ae/At) must be > 0.")
    if Pamb < 0:
        raise ValueError("Pamb must be >= 0 (Pa).")
    if eta_cf <= 0:
        raise ValueError("eta_cf must be > 0.")
    if nfz not in (0, 1, 2):
        raise ValueError("nfz must be one of {0 (equilibrium), 1 (frozen at throat), 2 (frozen at chamber)}.")

    # --- RocketCEA Cf lookup ---
    # RocketCEA returns a tuple; Cf is the middle value in these calls.
    if nfz == 0:
        _, cf, _ = cea_obj.get_PambCf(Pamb, Pc, MR, eps)
    else:
        frozenAtThroat = 0 if nfz == 1 else 1
        _, cf, _ = cea_obj.getFrozen_PambCf(Pamb, Pc, MR, eps, frozenAtThroat)

    # --- Thrust ---
    return float(eta_cf) * float(cf) * float(Pc) * float(At)



def get_chamber_pressure(target_density : float, MR : float, Pc_guess : float = 1e6, cea_obj : CEA_Obj = default_cea_obj):
    """
    Solve for chamber pressure using RocketCEA chamber density.

    This function inverts the RocketCEA relation:
        rho_c = get_Chamber_Density(Pc, MR)

    using a single initial guess (no bracketing).

    Parameters
    ----------
    cea_obj : CEA_Obj
        RocketCEA object configured with the desired fuel/oxidizer pair.
        Assumes all inputs/outputs are already in SI units.
    MR : float
        Mixture ratio (O/F) of the chamber.
    target_density : float
        Chamber density [kg/m^3].
    Pc_guess : float, optional
        Initial guess for chamber pressure [Pa].
        Default is 1e6 Pa (~10 bar).

    Returns
    -------
    Pc : float
        Chamber pressure [Pa].

    Raises
    ------
    ValueError
        If the root solve fails to converge.

    Notes
    -----
    - Uses a secant method (no bracketing required).
    - Convergence depends on a reasonable initial guess.
    - Best used in transients with Pc_guess = previous timestep pressure.
    """

    if target_density <= 0:
        raise ValueError("target_density must be > 0")
    if MR <= 0:
        raise ValueError("MR must be > 0")
    if Pc_guess <= 0:
        raise ValueError("Pc_guess must be > 0")

    def residual(log_Pc):
        Pc = np.exp(log_Pc)
        rho = cea_obj.get_Chamber_Density(Pc=Pc, MR=MR)
        return rho - target_density

    sol = root_scalar(
        residual,
        x0=np.log(Pc_guess),
        x1=np.log(Pc_guess * 1.05),
        method="secant",
    )

    if not sol.converged:
        raise ValueError(
            f"Chamber pressure solve failed. "
            f"Pc_guess={Pc_guess}, target_density={target_density}, MR={MR}"
        )

    return float(np.exp(sol.root))