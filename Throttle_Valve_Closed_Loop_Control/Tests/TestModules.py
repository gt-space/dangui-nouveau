import numpy as np
from rocketcea.cea_obj_w_units import CEA_Obj
from scipy.optimize import root, root_scalar
import matplotlib.pyplot as plt

PA_PER_PSI = 6894.757293168
N_PER_LBF = 4.4482216152605

# Assumes fully equilibrium for now, will change later
rocket = CEA_Obj(oxName='LOX', fuelName='RP-1',
                    temperature_units='degK', cstar_units='m/sec',
                    specific_heat_units='kJ/kg degK',
                    sonic_velocity_units='m/s', enthalpy_units='J/kg',
                    density_units='kg/m^3', pressure_units='Pa')


# Simple plot customizer
def set_winplot_dark():
    plt.rcParams.update({
        "figure.facecolor": "black",
        "axes.facecolor": "black",
        "savefig.facecolor": "black",
        "text.color": "white",
        "axes.labelcolor": "white",
        "axes.edgecolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "grid.color": "0.25",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "legend.frameon": True,
        "legend.facecolor": "black",
        "legend.edgecolor": "white",
    })


# TCA choked mdot (kg/s)
def get_mdot(Pc, mr, At, cstar_eff, cea_obj=rocket):
    cstar_ideal = cea_obj.get_Cstar(Pc, mr)     # m/s
    cstar = cstar_eff * cstar_ideal
    return (Pc * At) / cstar                   # kg/s

# Incompressible CdA equation
def mdot_from_cda(CdA, P1, P2, rho):
    dP = P1 - P2
    return np.sign(dP) * CdA * np.sqrt(2.0 * rho * np.abs(dP))

# Choked TCA thrust 
def get_thrust(Pc, mr, At, Pamb, eps, cf_eff=1, frozen=False, frozen_from_throat=False, cea_obj: CEA_Obj = rocket):
    if frozen:
        if frozen_from_throat:
            _, cf, _ = cea_obj.getFrozen_PambCf(Pamb, Pc, mr, eps, 1)
        else:
            _, cf, _ = cea_obj.getFrozen_PambCf(Pamb, Pc, mr, eps, 0)
    else:
        _, cf, _ = cea_obj.get_PambCf(Pamb, Pc, mr, eps)

    thrust = cf_eff * cf * Pc * At
    return thrust


# Solver Unknowns: [P_inj_fuel, P_inj_ox, Pc] all in Pa

def solve_pressure_ladder(
    fuel_tank_pressure, ox_tank_pressure,
    sys_fuel_CdA, sys_ox_CdA,
    inj_fuel_CdA, inj_ox_CdA,
    rho_fuel, rho_ox,
    throat_area, expansion_ratio,
    cstar_eff, cf_eff,
    ambient_pressure,
    cea_obj: CEA_Obj = rocket
):
    def residuals(x):
        P_inj_fuel, P_inj_ox, Pc = x  # Pa

        # Guard against weird pressures 
        if Pc <= 0 or P_inj_fuel <= 0 or P_inj_ox <= 0:
            return [1e9, 1e9, 1e9]

        # Plumbing mdots (kg/s)
        sys_mdot_f = mdot_from_cda(sys_fuel_CdA, fuel_tank_pressure, P_inj_fuel, rho_fuel)
        inj_mdot_f = mdot_from_cda(inj_fuel_CdA, P_inj_fuel, Pc, rho_fuel)

        sys_mdot_ox = mdot_from_cda(sys_ox_CdA, ox_tank_pressure, P_inj_ox, rho_ox)
        inj_mdot_ox = mdot_from_cda(inj_ox_CdA, P_inj_ox, Pc, rho_ox)

        # Account for flow reversal
        if inj_mdot_f <= 0 or inj_mdot_ox <= 0:
            return [1e6, 1e6, 1e6]

        mr = inj_mdot_ox / inj_mdot_f

        tca_mdot = get_mdot(Pc, mr, throat_area, cstar_eff, cea_obj)

        return [
            sys_mdot_f  - inj_mdot_f,                 # fuel continuity
            sys_mdot_ox - inj_mdot_ox,                # ox continuity
            (inj_mdot_ox + inj_mdot_f) - tca_mdot     # chamber continuity
        ]

    # Initial guesses (Pa)
    Pc0 = 0.60 * min(fuel_tank_pressure, ox_tank_pressure)
    x0 = np.array([
        0.80 * fuel_tank_pressure,   # P_inj_fuel
        0.80 * ox_tank_pressure,     # P_inj_ox
        Pc0                          # Pc
    ], dtype=float)

    sol = root(residuals, x0=x0, method="hybr")
    if not sol.success:
        raise RuntimeError(f"Pressure ladder solver failed: {sol.message}")

    P_inj_fuel, P_inj_ox, Pc = sol.x  # Pa

    # Injector stiffnesses
    stiff_f_pct  = (P_inj_fuel - Pc) / Pc * 100.0
    stiff_ox_pct = (P_inj_ox   - Pc) / Pc * 100.0

    inj_mdot_f  = mdot_from_cda(inj_fuel_CdA, P_inj_fuel, Pc, rho_fuel)
    inj_mdot_ox = mdot_from_cda(inj_ox_CdA,   P_inj_ox,   Pc, rho_ox)

    mr = float(inj_mdot_ox / inj_mdot_f)

    thrust = get_thrust(
        Pc=Pc,
        mr=mr,
        At=throat_area,
        Pamb=ambient_pressure,
        eps=expansion_ratio,
        cf_eff=cf_eff,
        cea_obj=cea_obj
    )  # N

    cstar = cea_obj.get_Cstar(Pc, mr) * cstar_eff  # m/s

    # Return SI: pressures (Pa), mdots (kg/s), thrust (N), cstar (m/s)
    return (float(P_inj_fuel),
            float(P_inj_ox),
            float(Pc),
            float(inj_mdot_f),
            float(inj_mdot_ox),
            mr,
            float(stiff_f_pct),
            float(stiff_ox_pct),
            float(thrust),
            float(cstar))



def solve_SysCdAs(Pc, MR,
                  P_tank_f, P_tank_ox,
                  inj_fuel_CdA, inj_ox_CdA,
                  rho_fuel, rho_ox,
                  throat_area, cstar_eff,
                  cea_obj: CEA_Obj = rocket):

    if Pc <= 0:
        raise ValueError("Pc must be > 0 (Pa).")
    if MR <= 0:
        raise ValueError("MR must be > 0 (O/F).")
    if inj_fuel_CdA <= 0 or inj_ox_CdA <= 0:
        raise ValueError("Injector CdA values must be > 0 (m^2).")
    if rho_fuel <= 0 or rho_ox <= 0:
        raise ValueError("Densities must be > 0 (kg/m^3).")
    if throat_area <= 0:
        raise ValueError("throat_area must be > 0 (m^2).")
    if cstar_eff <= 0:
        raise ValueError("cstar_eff must be > 0.")

    # choked mdot
    tca_mdot = get_mdot(Pc, MR, throat_area, cstar_eff, cea_obj)

    fuel_mdot = tca_mdot / (1.0 + MR)
    ox_mdot   = tca_mdot - fuel_mdot

    dP_fuel = (fuel_mdot / inj_fuel_CdA) ** 2 / (2.0 * rho_fuel)
    dP_ox   = (ox_mdot   / inj_ox_CdA)   ** 2 / (2.0 * rho_ox)

    P_inj_fuel = Pc + dP_fuel
    P_inj_ox = Pc + dP_ox


    if P_inj_fuel >= P_tank_f:
        raise RuntimeError("Fuel injector pressure exceeds fuel tank pressure.")
    if P_inj_ox >= P_tank_ox:
        raise RuntimeError("Ox injector pressure exceeds ox tank pressure.")

    sysCdA_fuel = fuel_mdot / np.sqrt(2*rho_fuel*(P_tank_f - P_inj_fuel))
    sysCdA_ox = ox_mdot / np.sqrt(2*rho_ox*(P_tank_ox - P_inj_ox))

    return (float(sysCdA_fuel),
            float(sysCdA_ox),
            float(P_inj_fuel), 
            float(P_inj_ox))


def solve_injector_pressures(Pc, MR,
                             inj_CdA_f, inj_CdA_ox,
                             rho_f, rho_ox,
                             throat_area, cstar_eff,
                             cea_obj: CEA_Obj = rocket
                             ):
    
    total_mdot = get_mdot(Pc, MR, throat_area, cstar_eff, cea_obj)
    fuel_mdot = (1/(1+MR)) * total_mdot
    ox_mdot = total_mdot - fuel_mdot

    P_inj_f = Pc + (fuel_mdot/inj_CdA_f)**2 / (2*rho_f)
    P_inj_ox = Pc + (ox_mdot/inj_CdA_ox)**2 / (2*rho_ox)

    return (P_inj_f, P_inj_ox)

