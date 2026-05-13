'''

INSTRUCTIONS TO RUN THIS FILE:

Becuase this file is not in the root directory and has relative imports
running it is not as easy just hitting run.

In the command terminal for the root directory (the overall project folder), run:

---> python3 -m Tests.test_single_controller

(replace 'python3' with just 'python' if you're not using Python 3)

'''

import copy
import numpy as np
import matplotlib.pyplot as plt

from Utilities import set_winplot_dark
from Utilities.Constants import PA_PER_PSI, M2_PER_IN2, LBF_PER_N
from Network.Components import *
from Network import TestStand, Balance
from Controller import TestActuator, PID, ramp

set_winplot_dark()


# ============================================================
# Time Grid
# ============================================================
dt = 0.001
timespan = np.arange(0.0, 10.0 + dt, dt)



# ============================================================
# Plant Setup
# ============================================================
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
    mixture_ratio=2.0,
    cstar_efficiency=1,
    volume=6e-2,
)

Atmosphere = Drain("Ambient")

FuelThrottle = Valve("Fuel Throttle Valve", CdA=0.5e-4)
OxThrottle = Valve("Oxidizer Throttle Valve", CdA=1.0e-4)

FuelRunline = Line("Fuel Runline", length=5, cross_sectional_area=0.5e-4)
OxRunline = Line("Oxidizer Runline", length=5, cross_sectional_area=0.5e-4)

FuelInjector = Orifice("Fuel Injector", 0.5e-4)
OxInjector = Orifice("OxInjector", 1.0e-4)

TCA = Nozzle(
    "Nozzle",
    throat_area=6.05 * M2_PER_IN2,
    contraction_ratio=2,
    expansion_ratio=4.7,
    eta_cf=0.95,
    nfz=2,
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


# --- Build Steady-State Initial Condition ---
ts = copy.deepcopy(HETS)

#ss_result = ts.steady_state()
MR_balance = Balance(
    tune="OxThrottleValve.CdA",
    measure="MainChamber.MR",
    target=2.3,
    bounds=(1e-6, 1e-4),
    tol=1e-5,   #
)
ss_result = ts.steady_state_with_balance(MR_balance)
if ss_result is not None:
    ts = ss_result

print(ts)


# ============================================================
# Actuator Setup
# ============================================================
ox_cda_initial = ts.OxThrottleValve.CdA
ox_cda_min = 1e-6
ox_cda_max = 3e-4
ox_cda_rate_limit = 0.75e-4

ox_actuator = TestActuator(
    initial_value=ox_cda_initial,
    min_value=ox_cda_min,
    max_value=ox_cda_max,
    max_rate=ox_cda_rate_limit,
)


# ============================================================
# Controller Setup
# ============================================================
pid = PID(
    Kp=5.0e-5,
    Ki=5.0e-5,
    Kd=0,
    u_min=ox_cda_min,
    u_max=ox_cda_max,
    u_bias=ox_cda_initial,
)
pid.reset(measurement=ts.MainChamber.MR)


# ============================================================
# Schedule Setup
# ============================================================
fuel_throttle_schedule = ramp(
    timespan,
    initial_value=ts.FuelThrottleValve.CdA,
    final_value=ts.FuelThrottleValve.CdA,
    t1=1.5,
    t2=3.0,
)

#mr_target_schedule = np.full_like(timespan, 2.0, dtype=float)
mr_target_schedule = ramp(timespan, 2.3, 2.0, 2, 4.0)



# ============================================================
# Storage
# ============================================================
fuel_sys_mdot = np.zeros(timespan.size)
ox_sys_mdot = np.zeros(timespan.size)

fuel_inj_pressure = np.zeros(timespan.size)
ox_inj_pressure = np.zeros(timespan.size)

fuel_inj_mdot = np.zeros(timespan.size)
ox_inj_mdot = np.zeros(timespan.size)

chamber_pressure = np.zeros(timespan.size)
mixture_ratio = np.zeros(timespan.size)
tca_mdot = np.zeros(timespan.size)
thrust = np.zeros(timespan.size)

fuel_throttle_cmd = np.zeros(timespan.size)
ox_throttle_cmd = np.zeros(timespan.size)
ox_throttle_unsat = np.zeros(timespan.size)

mr_error_hist = np.zeros(timespan.size)
mr_integral_hist = np.zeros(timespan.size)
mr_dmeas_hist = np.zeros(timespan.size)


# ============================================================
# Initial Storage
# ============================================================
fuel_sys_mdot[0] = ts.FuelRunline.mdot
ox_sys_mdot[0] = ts.OxRunline.mdot

fuel_inj_pressure[0] = ts.FuelInjectorManifold.p
ox_inj_pressure[0] = ts.OxInjectorManifold.p

fuel_inj_mdot[0] = ts.FuelInjector.mdot
ox_inj_mdot[0] = ts.OxInjector.mdot

chamber_pressure[0] = ts.MainChamber.p
mixture_ratio[0] = ts.MainChamber.MR
tca_mdot[0] = ts.TCA.mdot
thrust[0] = ts.TCA.F

fuel_throttle_cmd[0] = ts.FuelThrottleValve.CdA
ox_throttle_cmd[0] = ts.OxThrottleValve.CdA
ox_throttle_unsat[0] = ox_cda_initial

mr_error_hist[0] = mr_target_schedule[0] - ts.MainChamber.MR
mr_integral_hist[0] = pid.integral
mr_dmeas_hist[0] = 0.0


# ============================================================
# Closed-Loop Simulation
# ============================================================
for i, t in enumerate(timespan[:-1]):
    # Apply any scheduled fuel-side command
    ts.FuelThrottleValve.CdA = fuel_throttle_schedule[i]

    # Current measurement and target
    mr_measured = ts.MainChamber.MR
    mr_target = mr_target_schedule[i]

    # PID computes desired Ox throttle CdA
    ox_cmd, err, integ, dmeas = pid.update(
        target=mr_target,
        measurement=mr_measured,
        dt=dt,
    )

    # Store unsaturated value if the PID class exposes it; otherwise store command
    unsat_cmd = pid.u_unsat if hasattr(pid, "u_unsat") and pid.u_unsat is not None else ox_cmd
    ox_throttle_unsat[i + 1] = unsat_cmd

    # Actuator applies rate limiting and hard bounds
    ox_actual_cda = ox_actuator.update(ox_cmd, dt)
    ts.OxThrottleValve.CdA = ox_actual_cda

    '''
    if i % 200 == 0:
        print(
            f"t={t:.4f}, "
            f"MR={ts.MainChamber.MR:.4f}, "
            f"MR_target={mr_target:.4f}, "
            f"OxCdA={ts.OxThrottleValve.CdA:.6e}, "
            f"OxManP={ts.OxInjectorManifold.p:.3f}, "
            f"FuelManP={ts.FuelInjectorManifold.p:.3f}, "
            f"Pc={ts.MainChamber.p:.3f}"
        )

    if ts.OxInjectorManifold.p <= ts.MainChamber.p:
        print("Ox injector lost forward pressure drop before timestep")
        print("t =", t)
        print("Ox manifold p =", ts.OxInjectorManifold.p)
        print("Chamber p     =", ts.MainChamber.p)
        print("Ox throttle CdA =", ts.OxThrottleValve.CdA)
        break
    '''
    # Advance plant one timestep
    ts = ts.timestep(dt=dt)

    # Store histories
    fuel_sys_mdot[i + 1] = ts.FuelRunline.mdot
    ox_sys_mdot[i + 1] = ts.OxRunline.mdot

    fuel_inj_pressure[i + 1] = ts.FuelInjectorManifold.p
    ox_inj_pressure[i + 1] = ts.OxInjectorManifold.p

    fuel_inj_mdot[i + 1] = ts.FuelInjector.mdot
    ox_inj_mdot[i + 1] = ts.OxInjector.mdot

    chamber_pressure[i + 1] = ts.MainChamber.p
    mixture_ratio[i + 1] = ts.MainChamber.MR
    tca_mdot[i + 1] = ts.TCA.mdot
    thrust[i + 1] = ts.TCA.F

    fuel_throttle_cmd[i + 1] = ts.FuelThrottleValve.CdA
    ox_throttle_cmd[i + 1] = ts.OxThrottleValve.CdA

    mr_error_hist[i + 1] = err
    mr_integral_hist[i + 1] = integ
    mr_dmeas_hist[i + 1] = dmeas


print(ts)





# ============================================================
# Plotting
# ============================================================
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
    "thrust": "#FFA500",
}

control_colors = {
    "mr_error": "#39FF14",
    "mr_integral": "#FF00FF",
    "mr_dmeas": "#00BFFF",
    "ox_unsat": "#FF3131",
    "ox_actual": "#FFA500",
    "mr_target": "#FFD700",
}

fuel_sched_color = "#FFFFFF"
ox_sched_color = "#FFA500"

fuel_throttle_cm2 = fuel_throttle_cmd * 1e4
ox_throttle_cm2 = ox_throttle_cmd * 1e4
ox_throttle_unsat_cm2 = ox_throttle_unsat * 1e4

fuel_inj_pressure_psia = fuel_inj_pressure / PA_PER_PSI
ox_inj_pressure_psia = ox_inj_pressure / PA_PER_PSI
chamber_pressure_psia = chamber_pressure / PA_PER_PSI
thrust_lbf = thrust * LBF_PER_N


def add_throttle_overlay(ax):
    ax2 = ax.twinx()
    ax2.plot(
        timespan,
        fuel_throttle_cm2,
        color=fuel_sched_color,
        linestyle="--",
        linewidth=1.5,
        label="Fuel throttle",
    )
    ax2.plot(
        timespan,
        ox_throttle_cm2,
        color=ox_sched_color,
        linestyle="--",
        linewidth=1.5,
        label="Ox throttle",
    )
    ax2.set_ylabel("Throttle CdA (cm$^2$)")
    return ax2



thrust_lbf = thrust * 0.2248089431


# -------------------------
# Figure 1: Control
# -------------------------
fig_control, axs_control = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

axs_control[0].plot(
    timespan,
    mixture_ratio,
    color=chamber_colors["mr"],
    linewidth=2,
    label="Measured MR",
)
axs_control[0].plot(
    timespan,
    mr_target_schedule,
    color=control_colors["mr_target"],
    linestyle="--",
    linewidth=2,
    label="Target MR",
)
axs_control[0].set_ylabel("MR")
axs_control[0].set_title("Mixture Ratio Tracking")
axs_control[0].grid(alpha=0.3)
axs_control[0].legend()

axs_control[1].plot(
    timespan,
    mr_error_hist,
    color=control_colors["mr_error"],
    linewidth=2,
    label="MR error",
)
axs_control[1].plot(
    timespan,
    mr_integral_hist,
    color=control_colors["mr_integral"],
    linewidth=2,
    label="Integral state",
)
axs_control[1].plot(
    timespan,
    mr_dmeas_hist,
    color=control_colors["mr_dmeas"],
    linewidth=2,
    label="d(MR)/dt",
)
axs_control[1].set_ylabel("PID States")
axs_control[1].set_title("PID Internal States")
axs_control[1].grid(alpha=0.3)
axs_control[1].legend()

axs_control[2].plot(
    timespan,
    fuel_throttle_cm2,
    color=fuel_sched_color,
    linewidth=2,
    label="Fuel throttle CdA",
)
axs_control[2].plot(
    timespan,
    ox_throttle_cm2,
    color=control_colors["ox_actual"],
    linewidth=2,
    label="Ox throttle CdA",
)
axs_control[2].plot(
    timespan,
    ox_throttle_unsat_cm2,
    color=control_colors["ox_unsat"],
    linestyle="--",
    linewidth=1.75,
    label="Ox unsat command",
)
axs_control[2].set_ylabel("CdA (cm$^2$)")
axs_control[2].set_xlabel("Time (s)")
axs_control[2].set_title("Throttle Commands")
axs_control[2].grid(alpha=0.3)
axs_control[2].legend()

fig_control.tight_layout()


# -------------------------
# Figure 2: Fuel side
# -------------------------
fig_fuel, axs_fuel = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

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

fig_fuel.tight_layout()


# -------------------------
# Figure 3: Ox side
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
# Figure 4: Chamber / nozzle
# -------------------------
fig_chamber, axs_chamber = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axs_chamber = axs_chamber.flatten()

axs_chamber[0].plot(timespan, chamber_pressure_psia, color=chamber_colors["pressure"], linewidth=2)
axs_chamber[0].set_ylabel("Chamber P (psia)")
axs_chamber[0].set_title("Chamber Pressure")
axs_chamber[0].grid(alpha=0.3)
add_throttle_overlay(axs_chamber[0])

axs_chamber[1].plot(timespan, mixture_ratio, color=chamber_colors["mr"], linewidth=2, label="Measured MR")
axs_chamber[1].plot(timespan, mr_target_schedule, color=control_colors["mr_target"], linestyle="--", linewidth=2, label="Target MR")
axs_chamber[1].set_ylabel("Mixture Ratio")
axs_chamber[1].set_title("Mixture Ratio")
axs_chamber[1].grid(alpha=0.3)
axs_chamber[1].legend()
add_throttle_overlay(axs_chamber[1])

axs_chamber[2].plot(timespan, tca_mdot, color=chamber_colors["tca_mdot"], linewidth=2)
axs_chamber[2].set_ylabel("TCA mdot (kg/s)")
axs_chamber[2].set_xlabel("Time (s)")
axs_chamber[2].set_title("Nozzle Mass Flow")
axs_chamber[2].grid(alpha=0.3)
add_throttle_overlay(axs_chamber[2])

axs_chamber[3].plot(timespan, thrust_lbf, color=chamber_colors["thrust"], linewidth=2)
axs_chamber[3].set_ylabel("Thrust (lbf)")
axs_chamber[3].set_xlabel("Time (s)")
axs_chamber[3].set_title("Thrust")
axs_chamber[3].grid(alpha=0.3)
add_throttle_overlay(axs_chamber[3])

fig_chamber.tight_layout()

plt.show()