'''

INSTRUCTIONS TO RUN THIS FILE:

Becuase this file is not in the root directory and has relative imports
running it is not as easy just hitting run.

In the command terminal for the root directory (the overall project folder), run:

---> python3 -m Tests.test_transient_full

(replace 'python3' with just 'python' if you're not using Python 3)

'''



import copy
import numpy as np
import matplotlib.pyplot as plt

from Utilities import set_winplot_dark
from Utilities.Constants import PA_PER_PSI, M2_PER_IN2
from Network.Components import *
from Network import TestStand, Balance

set_winplot_dark()

FuelTank = Tank("Fuel Tank", "jet-a", 450 * PA_PER_PSI, 70 / 1000, temperature=300)
OxTank = Tank("Oxidizer Tank", "LOX", 400 * PA_PER_PSI, 70 / 1000, temperature=90)

FuelInjectorManifold = InjectorManifold(
    "Fuel Manifold",
    350 * PA_PER_PSI,
    temperature=500,
    volume=0.1287,
)
OxInjectorManifold = InjectorManifold(
    "Oxidizer Manifold",
    350 * PA_PER_PSI,
    temperature=90,
    volume=0.1287,
)

Chamber = CombustionChamber(
    "Main Chamber",
    300 * PA_PER_PSI,
    #mixture_ratio=2.4173,
    mixture_ratio=2,
    cstar_efficiency=0.8,
    volume=6e-2,
)

Atmosphere = Drain("Ambient")

FuelThrottle = Valve("Fuel Throttle Valve", CdA=0.5e-4)
OxThrottle = Valve("Oxidizer Throttle Valve", CdA=1.0e-4)

FuelRunline = Line("Fuel Runline", length=5, cross_sectional_area=0.5e-4)
OxRunline = Line("Oxidizer Runline", length=5, cross_sectional_area=0.5e-4)

FuelInjector = Orifice("Fuel Injector", 0.5e-4)
OxInjector = Orifice("Oxidizer Injector", 1.0e-4)

TCA = Nozzle(
    "Nozzle", 
    throat_area=6.05 * M2_PER_IN2, 
    contraction_ratio=2, 
    expansion_ratio=4.7, 
    eta_cf=0.95,
    nfz=2
)

HETS = TestStand(
    "HETS",
    FuelTank, OxTank,
    FuelRunline, OxRunline,
    FuelThrottle, OxThrottle,
    FuelInjectorManifold, OxInjectorManifold,
    FuelInjector, OxInjector,
    Chamber, TCA, Atmosphere
)


def ramp_hold_schedule(timespan, initial_value, final_value, t1, t2):
    """
    Generate a piecewise schedule that is:
    - constant at initial_value for t < t1
    - linearly ramped from initial_value to final_value for t1 <= t < t2
    - constant at final_value for t >= t2

    Parameters
    ----------
    timespan : np.ndarray
        1D array of time values.
    initial_value : float
        Value before the ramp starts.
    final_value : float
        Value after the ramp ends.
    t1 : float
        Ramp start time [s].
    t2 : float
        Ramp end time [s]. Must satisfy t2 > t1.

    Returns
    -------
    np.ndarray
        Array of scheduled values with the same shape as timespan.
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


# ============================================
# Time grid
# ============================================
dt = 0.001
timespan = np.arange(0, 10 + dt, dt)

# ============================================
# Build steady-state initial condition
# ============================================
ts = copy.deepcopy(HETS)

#ss_result = ts.steady_state()
MR_balance = Balance(
    tune="FuelInjector.CdA",
    measure="MainChamber.MR",
    target=2,
    bounds=(1e-6, 1e-4),
    tol=1e-5,   #
)
ss_result = ts.steady_state_with_balance(MR_balance)
if ss_result is not None:
    ts = ss_result
print(ts)

# ============================================
# Throttle schedules
# constant -> ramp -> constant
# ============================================
'''fuel_throttle_CdA = ramp_hold_schedule(
    timespan,
    initial_value=ts.FuelThrottleValve.CdA,
    final_value=2.0 * ts.FuelThrottleValve.CdA,
    t1=1.5,
    t2=3.0,
)

ox_throttle_CdA = ramp_hold_schedule(
    timespan,
    initial_value=ts.OxThrottleValve.CdA,
    final_value=1.15 * ts.OxThrottleValve.CdA,
    t1=1.5,
    t2=3.0,
)'''

fuel_throttle_CdA = ramp_hold_schedule(
    timespan,
    initial_value=ts.FuelThrottleValve.CdA,
    final_value=2.0 * ts.FuelThrottleValve.CdA,
    t1=1.5,
    t2=3.0,
)

ox_throttle_CdA = ramp_hold_schedule(
    timespan,
    initial_value=ts.OxThrottleValve.CdA,
    final_value=1.15 * ts.OxThrottleValve.CdA,
    t1=1.5,
    t2=2.5,
)

# ============================================
# Storage arrays for plotting / checking
# ============================================
fuel_sys_mdot = np.zeros(timespan.size)
ox_sys_mdot = np.zeros(timespan.size)

fuel_inj_pressure = np.zeros(timespan.size)
ox_inj_pressure = np.zeros(timespan.size)

fuel_inj_mdot = np.zeros(timespan.size)
ox_inj_mdot = np.zeros(timespan.size)

chamber_pressure = np.zeros(timespan.size)
mixture_ratio = np.zeros(timespan.size)
tca_mdot = np.zeros(timespan.size)

fuel_throttle_cmd = np.zeros(timespan.size)
ox_throttle_cmd = np.zeros(timespan.size)

# ============================================
# Initial condition storage from steady state
# ============================================
fuel_sys_mdot[0] = ts.FuelRunline.mdot
ox_sys_mdot[0] = ts.OxRunline.mdot

fuel_inj_pressure[0] = ts.FuelInjectorManifold.p
ox_inj_pressure[0] = ts.OxInjectorManifold.p

fuel_inj_mdot[0] = ts.FuelInjector.mdot
ox_inj_mdot[0] = ts.OxInjector.mdot

chamber_pressure[0] = ts.MainChamber.p
mixture_ratio[0] = ts.MainChamber.MR
tca_mdot[0] = ts.TCA.mdot

fuel_throttle_cmd[0] = ts.FuelThrottleValve.CdA
ox_throttle_cmd[0] = ts.OxThrottleValve.CdA

# ============================================
# March the TestStand forward in time
# ============================================
for i, _ in enumerate(timespan[:-1]):
    ts.FuelThrottleValve.CdA = fuel_throttle_CdA[i]
    ts.OxThrottleValve.CdA = ox_throttle_CdA[i]

    ts = ts.timestep(dt=dt)

    fuel_sys_mdot[i + 1] = ts.FuelRunline.mdot
    ox_sys_mdot[i + 1] = ts.OxRunline.mdot

    fuel_inj_pressure[i + 1] = ts.FuelInjectorManifold.p
    ox_inj_pressure[i + 1] = ts.OxInjectorManifold.p

    fuel_inj_mdot[i + 1] = ts.FuelInjector.mdot
    ox_inj_mdot[i + 1] = ts.OxInjector.mdot

    chamber_pressure[i + 1] = ts.MainChamber.p
    mixture_ratio[i + 1] = ts.MainChamber.MR
    tca_mdot[i + 1] = ts.TCA.mdot

    fuel_throttle_cmd[i + 1] = ts.FuelThrottleValve.CdA
    ox_throttle_cmd[i + 1] = ts.OxThrottleValve.CdA

print(ts)

# ============================================
# Plotting
# ============================================
fuel_colors = {
    "sys_mdot": "#00FFFF",
    "inj_pressure": "#FF00FF",
    "inj_mdot": "#39FF14",
}

ox_colors = {
    "sys_mdot": "#FF3131",
    "inj_pressure": "#00BFFF",
    "inj_mdot": "#FF6EC7",
}

chamber_colors = {
    "pressure": "#FFD700",
    "mr": "#39FF14",
    "tca_mdot": "#00FFFF",
}

fuel_sched_color = "#FFFFFF"
ox_sched_color = "#FFA500"

fuel_throttle_cm2 = fuel_throttle_cmd * 1e4
ox_throttle_cm2 = ox_throttle_cmd * 1e4

fuel_inj_pressure_psia = fuel_inj_pressure / PA_PER_PSI
ox_inj_pressure_psia = ox_inj_pressure / PA_PER_PSI
chamber_pressure_psia = chamber_pressure / PA_PER_PSI


def add_throttle_overlay(ax):
    ax2 = ax.twinx()
    ax2.plot(timespan, fuel_throttle_cm2, color=fuel_sched_color, linestyle="--", linewidth=1.5)
    ax2.plot(timespan, ox_throttle_cm2, color=ox_sched_color, linestyle="--", linewidth=1.5)
    ax2.set_ylabel("Throttle CdA (cm²)")
    return ax2


# -------------------------
# Figure 1: Fuel side
# -------------------------
fig_fuel, axs_fuel = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axs_fuel = axs_fuel.flatten()

axs_fuel[0].plot(timespan, fuel_sys_mdot, color=fuel_colors["sys_mdot"], linewidth=2)
axs_fuel[0].set_ylabel("Fuel System mdot (kg/s)")
axs_fuel[0].set_title("Fuel System Mass Flow")
axs_fuel[0].grid(alpha=0.3)
add_throttle_overlay(axs_fuel[0])

axs_fuel[1].plot(timespan, fuel_inj_pressure_psia, color=fuel_colors["inj_pressure"], linewidth=2)
axs_fuel[1].set_ylabel("Fuel Injector P (psia)")
axs_fuel[1].set_title("Fuel Injector Pressure")
axs_fuel[1].grid(alpha=0.3)
add_throttle_overlay(axs_fuel[1])

axs_fuel[2].plot(timespan, fuel_inj_mdot, color=fuel_colors["inj_mdot"], linewidth=2)
axs_fuel[2].set_ylabel("Fuel Injector mdot (kg/s)")
axs_fuel[2].set_xlabel("Time (s)")
axs_fuel[2].set_title("Fuel Injector Mass Flow")
axs_fuel[2].grid(alpha=0.3)
add_throttle_overlay(axs_fuel[2])

axs_fuel[3].plot(timespan, fuel_throttle_cm2, color=fuel_sched_color, linewidth=2, label="Fuel")
axs_fuel[3].plot(timespan, ox_throttle_cm2, color=ox_sched_color, linewidth=2, label="Ox")
axs_fuel[3].set_ylabel("Throttle CdA (cm²)")
axs_fuel[3].set_xlabel("Time (s)")
axs_fuel[3].set_title("Throttle Schedules")
axs_fuel[3].grid(alpha=0.3)
axs_fuel[3].legend()

fig_fuel.tight_layout()

# -------------------------
# Figure 2: Ox side
# -------------------------
fig_ox, axs_ox = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

axs_ox[0].plot(timespan, ox_sys_mdot, color=ox_colors["sys_mdot"], linewidth=2)
axs_ox[0].set_ylabel("Ox System mdot (kg/s)")
axs_ox[0].set_title("Oxidizer System Mass Flow")
axs_ox[0].grid(alpha=0.3)
add_throttle_overlay(axs_ox[0])

axs_ox[1].plot(timespan, ox_inj_pressure_psia, color=ox_colors["inj_pressure"], linewidth=2)
axs_ox[1].set_ylabel("Ox Injector P (psia)")
axs_ox[1].set_title("Oxidizer Injector Pressure")
axs_ox[1].grid(alpha=0.3)
add_throttle_overlay(axs_ox[1])

axs_ox[2].plot(timespan, ox_inj_mdot, color=ox_colors["inj_mdot"], linewidth=2)
axs_ox[2].set_ylabel("Ox Injector mdot (kg/s)")
axs_ox[2].set_xlabel("Time (s)")
axs_ox[2].set_title("Oxidizer Injector Mass Flow")
axs_ox[2].grid(alpha=0.3)
add_throttle_overlay(axs_ox[2])

fig_ox.tight_layout()

# -------------------------
# Figure 3: Chamber / nozzle
# -------------------------
fig_chamber, axs_chamber = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

axs_chamber[0].plot(timespan, chamber_pressure_psia, color=chamber_colors["pressure"], linewidth=2)
axs_chamber[0].set_ylabel("Chamber P (psia)")
axs_chamber[0].set_title("Chamber Pressure")
axs_chamber[0].grid(alpha=0.3)
add_throttle_overlay(axs_chamber[0])

axs_chamber[1].plot(timespan, mixture_ratio, color=chamber_colors["mr"], linewidth=2)
axs_chamber[1].set_ylabel("Mixture Ratio")
axs_chamber[1].set_title("Mixture Ratio")
axs_chamber[1].grid(alpha=0.3)
add_throttle_overlay(axs_chamber[1])

axs_chamber[2].plot(timespan, tca_mdot, color=chamber_colors["tca_mdot"], linewidth=2)
axs_chamber[2].set_ylabel("TCA mdot (kg/s)")
axs_chamber[2].set_xlabel("Time (s)")
axs_chamber[2].set_title("Nozzle Mass Flow")
axs_chamber[2].grid(alpha=0.3)
add_throttle_overlay(axs_chamber[2])

fig_chamber.tight_layout()

plt.show()