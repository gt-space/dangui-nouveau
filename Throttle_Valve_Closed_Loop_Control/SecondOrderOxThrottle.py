"""
Closed-loop Pc control using only the oxidizer throttle valve.

Block diagram
-------------
Pc_target_schedule
        |
        v
Pc_measured --> PID on Pc ----------> ox_cmd_target
        |                                  |
        |                           second-order
        |                         actuator dynamics
        |                                  v
        |                           ox actuator state
        |                           (CdA, CdA rate)
        |                                  |
        +----------------------------------+
                                      |
                                      v
                           Ox throttle actuator
                                      |
                                      v
                                    plant
                                      |
                                      v
                             Pc, MR, mdot, thrust

Notes
-----
- Only the oxidizer throttle is actively controlled.
- The fuel throttle may still be scheduled independently to create disturbances
  or operating-point changes, but it is not part of the feedback loop.
"""

# ============================================================
# Controller Limits and Actuator Dynamics
# ============================================================
# This controller uses a feedforward + PID trim structure:
#   ox_cmd = ox_ff + delta_ox
#
# The remaining limits each serve a specific purpose:
#
# - Dynamic PID bounds:
#     pid_pc.u_min = ox_cmd_min - ox_ff
#     pid_pc.u_max = ox_cmd_max - ox_ff
#   These ensure the PID output (delta_ox) cannot push the
#   total Ox valve command outside physical valve limits.
#
# - Final command clipping:
#     ox_cmd_target = clip(ox_cmd_target, min, max)
#   This is a final safety check to guarantee a valid valve command.
#
# - Actuator dynamics (in SecondOrderActuator):
#   These represent the real valve hardware more realistically:
#     • minimum and maximum CdA (valve cannot exceed physical bounds)
#     • second-order response with natural frequency wn
#     • damping ratio zeta
#     • optional hard limit on CdA rate via max_velocity
#
# - Optional filtering (tau_pc, tau_d):
#   These smooth noisy signals:
#     • tau_pc filters measured chamber pressure before PID
#     • tau_d filters the derivative term (dPc/dt) inside the PID
#
# ------------------------------------------------------------
# Future extensions (if needed):
# You can reintroduce additional smoothing layers such as:
#   • feedforward rate limiting (smooth ox_ff changes)
#   • PID output slew rate limiting (limit how fast delta_ox changes)
#   • multi-rate simulation (faster actuator loop than outer Pc loop)
# These can help reduce oscillations or aggressive commands.
# ============================================================

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Utilities import set_winplot_dark
from Utilities.Constants import PA_PER_PSI, LBF_PER_N
from Network import Balance
from Controller import SecondOrderActuator, PID, ramp, step, low_pass_filter
from HETS import HETS

set_winplot_dark()  # cool plots format


# ============================================================
# User Inputs / Tunables
# ============================================================

# --- Simulation time grid ---
dt = 0.01                                  # controller / plant timestep [s]
t_final = 10.0                             # total simulation time [s]

# --- Plant Test Stand ---
test_stand = HETS                          # Choose what test stand you want as a plant (probably HETS)
print_initial_condition = True
print_final_condition = True

# --- Initial-condition setup ---
use_initial_balance = True                 # if False, use ts.steady_state() directly (balance can be edited below)

# --- Ox actuator travel limits ---
ox_cmd_min = 2.2e-5                        # minimum commanded CdA [m^2]
ox_cmd_max = 1.0e-3                        # maximum commanded CdA [m^2]

# --- Ox second-order actuator parameters ---
ox_actuator_wn = 80.0                      # natural frequency [rad/s]
ox_actuator_zeta = 0.6                     # damping ratio [-]
ox_actuator_initial_velocity = 0.0         # initial CdA rate [m^2/s]
ox_actuator_max_velocity = 1.5e-4          # optional max CdA rate [m^2/s]; None disables explicit rate clipping

# --- Ox map input ---
oxCdA_map_filename = "oxCdA_map.parquet"   # steady-state map file: Ox Throttle CdA -> state vector

# --- Pressure command schedule ---
# DEFINED BELOW

# --- Fuel throttle schedule ---
# DEFINED BELOW

# --- Pressure controller tuning ---
Kp_pc = 5.0e-10                            # proportional gain on Pc error
Ki_pc = 5.0e-10                             # integral gain on Pc error
Kd_pc = 2.0e-10                                # derivative gain on Pc error

u_bias_pc = 0.0                            # baseline valve command; if None, initialize to current steady-state CdA
tau_d_pc = 0.0                             # derivative filter time constant [s]

# --- Additional command shaping ---
tau_pc = 0.0                               # optional low-pass filter on measured Pc [s]


# ============================================================
# Time Grid
# ============================================================
timespan = np.arange(0.0, t_final + dt, dt)


# ============================================================
# Plant Setup + Build Steady-State Initial Condition
# ============================================================
ts = copy.deepcopy(test_stand)

initial_balance = Balance(
    tune="OxThrottleValve.CdA",
    measure="MainChamber.MR",
    target=2.0,
    bounds=(2.2e-5, 1e-4),
    tol=1e-5,
)

if use_initial_balance:
    ss_result = ts.steady_state_with_balance(initial_balance)
else:
    ss_result = ts.steady_state()

if ss_result is not None:
    ts = ss_result

if print_initial_condition:
    print(ts)


# ============================================================
# Actuator Setup
# ============================================================
ox_cda_initial = ts.OxThrottleValve.CdA

ox_actuator = SecondOrderActuator(
    initial_value=ox_cda_initial,              # actuator starts from actual plant valve state
    min_value=ox_cmd_min,
    max_value=ox_cmd_max,
    wn=ox_actuator_wn,
    zeta=ox_actuator_zeta,
    initial_velocity=ox_actuator_initial_velocity,
    max_velocity=ox_actuator_max_velocity,
)

extreme = copy.deepcopy(HETS)
result = extreme.check_max_throttlable_condition(
    max_fuel_CdA=ts.FuelThrottleValve.CdA,
    max_ox_CdA=ox_cmd_max,
)

print("\n========== MAX THROTTLE CONDITION ==========\n")
print(f"{'FIXED FUEL / MAX OX RANGE':>25}")
print(f"  Fuel CdA : {result['fuel_CdA']:.4e} m^2")
print(f"  Ox CdA   : {result['ox_CdA']:.4e} m^2")
print(f"  Pc       : {result['Pc'] / PA_PER_PSI:8.2f} psia")
print(f"  MR       : {result['MR']:.4f}")
print("-" * 40)


# ============================================================
# Load Ox CdA Map
# ============================================================
df = pd.read_parquet(oxCdA_map_filename)   # expected columns: fuel_CdA, ox_CdA, Pc, MR, ...

Pc_vec = df["Pc"].values
ox_CdA_vec = df["ox_CdA"].values


# ============================================================
# Schedule Setup
# ============================================================
Pc_initial = ts.MainChamber.p

Pc_target_schedule = ramp(
    timespan,
    initial_value=Pc_initial,
    final_value=280 * PA_PER_PSI,
    t1=0.0,
    t2=1.0,
)

Pc_target_schedule = step(
    timespan,
    ts.MainChamber.p,
    300*PA_PER_PSI,
    0
)

fuel_throttle_schedule = ramp(
    timespan,
    initial_value=ts.FuelThrottleValve.CdA,
    final_value=ts.FuelThrottleValve.CdA,
    t1=1.5,
    t2=3.0,
)


# ============================================================
# Pressure Controller (Pc loop)
# ============================================================
if u_bias_pc is None:
    u_bias_pc = ox_cda_initial

pid_pc = PID(
    Kp=Kp_pc,
    Ki=Ki_pc,
    Kd=Kd_pc,
    u_min=-1.0,                    # placeholder (overwritten every loop)
    u_max=1.0,
    u_bias=u_bias_pc,
    tau_d=tau_d_pc,
    du_dt_limit=None,              # explicitly disable
)

pid_pc.reset(measurement=ts.MainChamber.p, output=0.0)


# ============================================================
# Storage
# ============================================================
fuel_sys_mdot = np.zeros_like(timespan)
ox_sys_mdot = np.zeros_like(timespan)

fuel_inj_pressure = np.zeros_like(timespan)
ox_inj_pressure = np.zeros_like(timespan)

fuel_inj_mdot = np.zeros_like(timespan)
ox_inj_mdot = np.zeros_like(timespan)

chamber_pressure = np.zeros_like(timespan)
mixture_ratio = np.zeros_like(timespan)
tca_mdot = np.zeros_like(timespan)
thrust = np.zeros_like(timespan)

fuel_throttle_cmd = np.zeros_like(timespan)
ox_throttle_cmd = np.zeros_like(timespan)
ox_throttle_unsat = np.zeros_like(timespan)
ox_ff_cmd = np.zeros_like(timespan)
ox_actuator_velocity = np.zeros_like(timespan)

Pc_error_hist = np.zeros_like(timespan)
Pc_integral_hist = np.zeros_like(timespan)
Pc_dmeas_hist = np.zeros_like(timespan)


# ============================================================
# Initial Storage
# ============================================================
fuel_sys_mdot[0] = ts.FuelRunline.mdot
ox_sys_mdot[0] = ts.OxRunline.mdot

fuel_inj_pressure[0] = ts.FuelInjectorManifold.p
ox_inj_pressure[0] = ts.OxInjectorManifold.p

fuel_inj_mdot[0] = ts.FuelInjector.mdot
ox_inj_mdot[0] = ts.OxInjector.mdot

Pc_filt = ts.MainChamber.p
chamber_pressure[0] = ts.MainChamber.p
mixture_ratio[0] = ts.MainChamber.MR
tca_mdot[0] = ts.TCA.mdot
thrust[0] = ts.TCA.F

fuel_throttle_cmd[0] = ts.FuelThrottleValve.CdA
ox_throttle_cmd[0] = ts.OxThrottleValve.CdA
ox_ff_cmd[0] = np.interp(Pc_target_schedule[0], Pc_vec, ox_CdA_vec)
ox_throttle_unsat[0] = ox_ff_cmd[0]
ox_actuator_velocity[0] = ox_actuator.velocity

Pc_error_hist[0] = Pc_target_schedule[0] - ts.MainChamber.p
Pc_integral_hist[0] = pid_pc.integral
Pc_dmeas_hist[0] = 0.0


# ============================================================
# Closed-Loop Simulation
# ============================================================
for i, t in enumerate(timespan[:-1]):

    ts.FuelThrottleValve.CdA = fuel_throttle_schedule[i]   # apply scheduled fuel command
    Pc_raw = ts.MainChamber.p

    if tau_pc > 0.0:
        Pc_filt = low_pass_filter(Pc_filt, Pc_raw, dt, tau_pc)
    else:
        Pc_filt = Pc_raw

    Pc_measured = Pc_filt
    Pc_target = Pc_target_schedule[i]

    # Feedforward from steady-state Ox CdA -> Pc map
    ox_ff_target = np.interp(Pc_target, Pc_vec, ox_CdA_vec)
    ox_ff = np.clip(ox_ff_target, ox_cmd_min, ox_cmd_max)

    # Dynamic PID bounds
    pid_pc.u_min = ox_cmd_min - ox_ff
    pid_pc.u_max = ox_cmd_max - ox_ff
    pid_pc.u_bias = 0.0

    # PID update
    delta_ox, err_pc, integ_pc, dmeas_pc = pid_pc.update(
        target=Pc_target,
        measurement=Pc_measured,
        dt=dt,
    )

    # Combine FF + feedback
    ox_cmd_target = ox_ff + delta_ox
    ox_cmd_target = np.clip(ox_cmd_target, ox_cmd_min, ox_cmd_max)

    ox_ff_cmd[i + 1] = ox_ff
    ox_throttle_unsat[i + 1] = ox_cmd_target

    # Second-order actuator
    ox_cmd = ox_actuator.update(ox_cmd_target, dt)
    ts.OxThrottleValve.CdA = ox_cmd

    if ts.OxInjectorManifold.p <= ts.MainChamber.p:
        print(
            f"[WARN] t={t:.4f} s, "
            f"P_ox_man={ts.OxInjectorManifold.p:.6e} Pa, "
            f"Pc={ts.MainChamber.p:.6e} Pa, "
            f"ox_ff={ox_ff:.6e}, "
            f"delta_ox={delta_ox:.6e}, "
            f"ox_cmd_target={ox_cmd_target:.6e}, "
            f"ox_cmd={ox_cmd:.6e}, "
            f"ox_cmd_dot={ox_actuator.velocity:.6e}"
        )

    # Advance plant one timestep
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
    thrust[i + 1] = ts.TCA.F

    fuel_throttle_cmd[i + 1] = ts.FuelThrottleValve.CdA
    ox_throttle_cmd[i + 1] = ts.OxThrottleValve.CdA
    ox_actuator_velocity[i + 1] = ox_actuator.velocity

    Pc_error_hist[i + 1] = err_pc
    Pc_integral_hist[i + 1] = integ_pc
    Pc_dmeas_hist[i + 1] = dmeas_pc


if print_final_condition:
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
    "pc_error": "#00BFFF",
    "pc_integral": "#FF6EC7",
    "pc_dmeas": "#00FFFF",
    "ox_unsat": "#FF3131",
    "ox_actual": "#FFA500",
    "pc_target": "#FFD700",
    "ox_velocity": "#39FF14",
    "ox_ff": "#FFFFFF",
}

fuel_sched_color = "#FFFFFF"
ox_sched_color = "#FFA500"

fuel_throttle_cm2 = fuel_throttle_cmd * 1e4
ox_throttle_cm2 = ox_throttle_cmd * 1e4
ox_throttle_unsat_cm2 = ox_throttle_unsat * 1e4
ox_ff_cmd_cm2 = ox_ff_cmd * 1e4
ox_actuator_velocity_cm2s = ox_actuator_velocity * 1e4

fuel_inj_pressure_psia = fuel_inj_pressure / PA_PER_PSI
ox_inj_pressure_psia = ox_inj_pressure / PA_PER_PSI
chamber_pressure_psia = chamber_pressure / PA_PER_PSI
Pc_target_psia = Pc_target_schedule / PA_PER_PSI
thrust_lbf = thrust * LBF_PER_N


def add_fuel_throttle_overlay(ax):
    ax2 = ax.twinx()
    ax2.plot(
        timespan,
        fuel_throttle_cm2,
        color=fuel_sched_color,
        linestyle="--",
        linewidth=1.5,
        label="Fuel throttle",
    )
    ax2.set_ylabel("Fuel Throttle CdA (cm$^2$)")
    return ax2


def add_ox_throttle_overlay(ax):
    ax2 = ax.twinx()
    ax2.plot(
        timespan,
        ox_throttle_cm2,
        color=ox_sched_color,
        linestyle="--",
        linewidth=1.5,
        label="Ox throttle",
    )
    ax2.set_ylabel("Ox Throttle CdA (cm$^2$)")
    return ax2


# -------------------------
# Figure 1: Control loop
# -------------------------
fig_control, axs_control = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

axs_control[0].plot(
    timespan,
    chamber_pressure_psia,
    color=chamber_colors["pressure"],
    linewidth=2,
    label="Measured Pc",
)
axs_control[0].plot(
    timespan,
    Pc_target_psia,
    color=control_colors["pc_target"],
    linestyle="--",
    linewidth=2,
    label="Target Pc",
)
axs_control[0].set_ylabel("Pc (psia)")
axs_control[0].set_title("Chamber Pressure Tracking")
axs_control[0].grid(alpha=0.3)
axs_control[0].legend()

axs_control[1].plot(
    timespan,
    Pc_error_hist,
    color=control_colors["pc_error"],
    linewidth=2,
    label="Pc error",
)
axs_control[1].plot(
    timespan,
    Pc_integral_hist,
    color=control_colors["pc_integral"],
    linewidth=2,
    label="Pc integral",
)
axs_control[1].plot(
    timespan,
    Pc_dmeas_hist,
    color=control_colors["pc_dmeas"],
    linewidth=2,
    label="d(Pc)/dt term",
)
axs_control[1].set_ylabel("Pc Loop State")
axs_control[1].set_title("Pressure PID Internal States")
axs_control[1].grid(alpha=0.3)
axs_control[1].legend()

axs_control[2].plot(
    timespan,
    ox_ff_cmd_cm2,
    color=control_colors["ox_ff"],
    linewidth=1.75,
    label="Ox feedforward",
)
axs_control[2].plot(
    timespan,
    ox_throttle_cm2,
    color=control_colors["ox_actual"],
    linewidth=2,
    label="Ox actual CdA",
)
axs_control[2].plot(
    timespan,
    ox_throttle_unsat_cm2,
    color=control_colors["ox_unsat"],
    linestyle="--",
    linewidth=1.75,
    label="Ox target command",
)
axs_control[2].plot(
    timespan,
    fuel_throttle_cm2,
    color=fuel_sched_color,
    linewidth=2,
    label="Fuel throttle CdA",
)
axs_control[2].set_ylabel("CdA (cm$^2$)")
axs_control[2].set_title("Throttle Commands")
axs_control[2].grid(alpha=0.3)
axs_control[2].legend()

axs_control[3].plot(
    timespan,
    ox_actuator_velocity_cm2s,
    color=control_colors["ox_velocity"],
    linewidth=2,
    label="Ox actuator CdA rate",
)
axs_control[3].set_ylabel("CdA rate (cm$^2$/s)")
axs_control[3].set_xlabel("Time (s)")
axs_control[3].set_title("Second-Order Actuator Velocity")
axs_control[3].grid(alpha=0.3)
axs_control[3].legend()

fig_control.tight_layout()


# -------------------------
# Figure 2: Fuel side
# -------------------------
fig_fuel, axs_fuel = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

axs_fuel[0].plot(timespan, fuel_sys_mdot, color=fuel_colors["sys_mdot"], linewidth=2)
axs_fuel[0].set_ylabel("Fuel System mdot (kg/s)")
axs_fuel[0].set_title("Fuel System Mass Flow")
axs_fuel[0].grid(alpha=0.3)
add_fuel_throttle_overlay(axs_fuel[0])

axs_fuel[1].plot(timespan, fuel_inj_pressure_psia, color=fuel_colors["inj_pressure"], linewidth=2)
axs_fuel[1].set_ylabel("Fuel Injector P (psia)")
axs_fuel[1].set_title("Fuel Injector Pressure")
axs_fuel[1].grid(alpha=0.3)
add_fuel_throttle_overlay(axs_fuel[1])

axs_fuel[2].plot(timespan, fuel_inj_mdot, color=fuel_colors["inj_mdot"], linewidth=2)
axs_fuel[2].set_ylabel("Fuel Injector mdot (kg/s)")
axs_fuel[2].set_xlabel("Time (s)")
axs_fuel[2].set_title("Fuel Injector Mass Flow")
axs_fuel[2].grid(alpha=0.3)
add_fuel_throttle_overlay(axs_fuel[2])

fig_fuel.tight_layout()


# -------------------------
# Figure 3: Ox side
# -------------------------
fig_ox, axs_ox = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

axs_ox[0].plot(timespan, ox_sys_mdot, color=ox_colors["sys_mdot"], linewidth=2)
axs_ox[0].set_ylabel("Ox System mdot (kg/s)")
axs_ox[0].set_title("Oxidizer System Mass Flow")
axs_ox[0].grid(alpha=0.3)
add_ox_throttle_overlay(axs_ox[0])

axs_ox[1].plot(timespan, ox_inj_pressure_psia, color=ox_colors["inj_pressure"], linewidth=2)
axs_ox[1].set_ylabel("Ox Injector P (psia)")
axs_ox[1].set_title("Oxidizer Injector Pressure")
axs_ox[1].grid(alpha=0.3)
add_ox_throttle_overlay(axs_ox[1])

axs_ox[2].plot(timespan, ox_inj_mdot, color=ox_colors["inj_mdot"], linewidth=2)
axs_ox[2].set_ylabel("Ox Injector mdot (kg/s)")
axs_ox[2].set_title("Oxidizer Injector Mass Flow")
axs_ox[2].grid(alpha=0.3)
add_ox_throttle_overlay(axs_ox[2])

axs_ox[3].plot(timespan, ox_actuator_velocity_cm2s, color=control_colors["ox_velocity"], linewidth=2)
axs_ox[3].set_ylabel("CdA rate (cm$^2$/s)")
axs_ox[3].set_xlabel("Time (s)")
axs_ox[3].set_title("Ox Actuator CdA Rate")
axs_ox[3].grid(alpha=0.3)

fig_ox.tight_layout()


# -------------------------
# Figure 4: Chamber / nozzle
# -------------------------
fig_chamber, axs_chamber = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

axs_chamber[0].plot(
    timespan,
    chamber_pressure_psia,
    color=chamber_colors["pressure"],
    linewidth=2,
    label="Measured Pc",
)
axs_chamber[0].plot(
    timespan,
    Pc_target_psia,
    color=control_colors["pc_target"],
    linestyle="--",
    linewidth=2,
    label="Target Pc",
)
axs_chamber[0].set_ylabel("Chamber P (psia)")
axs_chamber[0].set_title("Chamber Pressure")
axs_chamber[0].grid(alpha=0.3)
axs_chamber[0].legend()

axs_chamber[1].plot(
    timespan,
    mixture_ratio,
    color=chamber_colors["mr"],
    linewidth=2,
    label="Measured MR",
)
axs_chamber[1].set_ylabel("Mixture Ratio")
axs_chamber[1].set_title("Mixture Ratio")
axs_chamber[1].grid(alpha=0.3)
axs_chamber[1].legend()

axs_chamber[2].plot(timespan, tca_mdot, color=chamber_colors["tca_mdot"], linewidth=2)
axs_chamber[2].set_ylabel("TCA mdot (kg/s)")
axs_chamber[2].set_title("Nozzle Mass Flow")
axs_chamber[2].grid(alpha=0.3)

axs_chamber[3].plot(timespan, thrust_lbf, color=chamber_colors["thrust"], linewidth=2)
axs_chamber[3].set_ylabel("Thrust (lbf)")
axs_chamber[3].set_xlabel("Time (s)")
axs_chamber[3].set_title("Thrust")
axs_chamber[3].grid(alpha=0.3)

fig_chamber.tight_layout()

plt.show()