def run_DualThrottle():
    """
    Closed-loop Pc control using an alpha-based steady-state map.

    Block diagram
    -------------
    Pc_target_schedule
            |
            v
    Pc -> alpha steady-state map  -----> alpha_ff_target
            |                                  |
            |                             rate limit
            |                                  v
            |                              alpha_ff
            |                                  |
    Pc_measured --> PID on Pc ----------> delta_alpha
            |                                  |
            +----------------------------------+
                        alpha_cmd = alpha_ff + delta_alpha
                                        |
                                        v
                            alpha -> (fuel_CdA, ox_CdA)
                                        |
                                        v
                            actuator rate limits / saturations
                                        |
                                        v
                                        plant
                                        |
                                        v
                                Pc, MR, mdot, thrust
    """
    # ============================================================
    # Controller Limits and Rates 
    # ============================================================
    # This controller uses a feedforward + PID trim architecture:
    #   alpha_cmd = alpha_ff + delta_alpha
    #
    # The limits serve distinct purposes:
    #
    # - Dynamic PID bounds:
    #     pid_alpha.u_min = -alpha_ff
    #     pid_alpha.u_max = 1 - alpha_ff
    #   These ensure the PID correction (delta_alpha) cannot push the
    #   total alpha command outside the valid range [0, 1].
    #
    # - Final alpha clipping:
    #     alpha_cmd = clip(alpha_cmd, 0, 1)
    #   This is a final safety check to guarantee a valid command.
    #
    # - Actuator limits (in FirstOrderActuator):
    #   These enforce real physical constraints:
    #     • minimum and maximum CdA
    #     • maximum rate of change of CdA (valve cannot move instantly)
    #
    # - Optional filtering (tau_pc, tau_d):
    #   These smooth noisy signals:
    #     • tau_pc filters measured chamber pressure
    #     • tau_d filters the derivative term in the PID
    #
    # ------------------------------------------------------------
    # Future extensions (if needed):
    # You can introduce additional smoothing/limits such as:
    #   • feedforward rate limiting (smooth alpha_ff changes)
    #   • PID output slew rate limiting (limit how fast delta_alpha changes)
    # These are useful if you see aggressive command jumps or oscillations,
    # but are not required for basic operation.
    # ============================================================

    import copy
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from Utilities import set_winplot_dark
    from Utilities.Constants import PA_PER_PSI, LBF_PER_N
    from Network.Components import Source, Drain, Volume, Branch, Tank, InjectorManifold, CombustionChamber, Orifice, Line, Valve, Nozzle
    from Network import Balance
    from Controller import FirstOrderActuator, PID, ramp, low_pass_filter, step
    from HETS import HETS

    import os

    set_winplot_dark() # cool plots format



    # ============================================================
    # User Inputs / Tunables
    # ============================================================

    # --- Simulation time grid ---
    dt = 0.01                                  # controller / plant timestep [s]
    t_final = 10.0                             # total simulation time [s]

    # --- Plant Test Stand ---
    test_stand = HETS                          # Choose what test stand you want as a plant (probably HETS)
    print_initial_condition = False 
    print_final_condition = False            

    # --- Initial-condition setup ---
    use_initial_balance = True                 # if False, use ts.steady_state() directly (balance can be edited below) 

    # --- Actuator limits ---
    fuel_cmd_min = 1e-6                        # minimum commanded value [units]
    fuel_cmd_max = 1.5e-4                      # maximum commanded value [units]
    fuel_cmd_rate_limit = 1.5e-4               # max actuator rate [units/s]

    ox_cmd_min = 1e-6                          # minimum commanded value [units]
    ox_cmd_max = 1.5e-4                        # maximum commanded value [units]
    ox_cmd_rate_limit = 1.5e-4                 # max actuator rate [units/s]


    # --- Alpha map input ---
    alpha_map_filename = os.path.join(os.path.dirname(__file__), "alpha_map.parquet")   # steady-state map file: alpha -> state vector

    # --- Pressure command schedule ---
    # DEFINED BELOW

    # --- Alpha controller tuning ---
    Kp_alpha = 5.0e-8                          # proportional gain on Pc error
    Ki_alpha = 1.0e-6                          # integral gain on Pc error
    Kd_alpha = 0                               # derivative gain on Pc error
    u_bias_alpha = 0.0                         # PID output bias; keep zero because alpha_ff provides feedforward (what alpha should PID immediately try to jump to every dt)
    tau_d_alpha = 0.0                          # derivative filter time constant [s] (low pass filter on d_error / dt since the derivative can be noisy)

    # --- Additional command shaping ---
    tau_pc = 0.0                               # optional low-pass filter on Pc measurement [s] (could simulate an actual low pass filter on CHPT)





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
        target=2,
        bounds=(1e-6, 1e-4),
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
        
    initial_condition_ts = ts


    # ============================================================
    # Actuator Setup
    # ============================================================
    fuel_cda_initial = ts.FuelThrottleValve.CdA
    ox_cda_initial = ts.OxThrottleValve.CdA

    fuel_actuator = FirstOrderActuator(
        initial_value=fuel_cda_initial,        # actuator starts from actual plant valve state
        min_value=fuel_cmd_min,
        max_value=fuel_cmd_max,
        max_rate=fuel_cmd_rate_limit,
    )
    ox_actuator = FirstOrderActuator(
        initial_value=ox_cda_initial,          # actuator starts from actual plant valve state
        min_value=ox_cmd_min,
        max_value=ox_cmd_max,
        max_rate=ox_cmd_rate_limit,
    )

    extreme = copy.deepcopy(HETS)
    result = extreme.check_max_throttlable_condition(
        max_fuel_CdA=fuel_cmd_max,
        max_ox_CdA=ox_cmd_max,
    )

    print("\n========== MAX THROTTLE CONDITION ==========\n")
    print(f"{'MAX FUEL & MAX OX':>25}")
    print(f"  Fuel CdA : {result['fuel_CdA']:.4e} m^2")
    print(f"  Ox CdA   : {result['ox_CdA']:.4e} m^2")
    print(f"  Pc       : {result['Pc'] / PA_PER_PSI:8.2f} psia")
    print(f"  MR       : {result['MR']:.4f}")
    print("-" * 40)


    # ============================================================
    # Load Alpha Map
    # ============================================================
    df = pd.read_parquet(alpha_map_filename)   # alpha -> [Pc, MR, fuel_CdA, ox_CdA, mdot_f, mdot_ox]

    alpha_vec = df["alpha"].values
    Pc_vec = df["Pc"].values
    fuel_CdA_vec = df["fuel_CdA"].values
    ox_CdA_vec = df["ox_CdA"].values


    # ============================================================
    # Schedule Setup
    # ============================================================
    Pc_initial = ts.MainChamber.p

    Pc_target_schedule = ramp(
        timespan,
        initial_value=Pc_initial,
        final_value=300 * PA_PER_PSI,
        t1=0.0,
        t2=1.0,
    )

    # Alternative examples:
    '''
    Pc_target_schedule = step(
        timespan,
        initial_value=Pc_initial,
        final_value=300 * PA_PER_PSI,
        t_step=2.0,
    )
    '''

    '''
    Pc_target_schedule = np.full_like(timespan, Pc_initial)
    '''


    # ============================================================
    # Alpha Controller (Pc loop)
    # ============================================================
    pid_alpha = PID(
        Kp=Kp_alpha,
        Ki=Ki_alpha,
        Kd=Kd_alpha,
        u_min=-1.0,                         # temporary placeholder; overwritten every loop by dynamic alpha bounds
        u_max=1.0,                          # temporary placeholder; overwritten every loop by dynamic alpha bounds
        u_bias=u_bias_alpha,
        tau_d=tau_d_alpha,
        du_dt_limit=None,                   # disable PID output slew limiting
    )


    pid_alpha.reset(measurement=ts.MainChamber.p, output=0.0)  # start with zero feedback correction


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
    fuel_throttle_unsat = np.zeros_like(timespan)
    ox_throttle_unsat = np.zeros_like(timespan)

    Pc_error_hist = np.zeros_like(timespan)
    Pc_integral_hist = np.zeros_like(timespan)
    Pc_dmeas_hist = np.zeros_like(timespan)

    alpha_ff_hist = np.zeros_like(timespan)
    delta_alpha_hist = np.zeros_like(timespan)
    alpha_cmd_hist = np.zeros_like(timespan)


    # ============================================================
    # Initial Storage
    # ============================================================
    fuel_sys_mdot[0] = ts.FuelRunline.mdot
    ox_sys_mdot[0] = ts.OxRunline.mdot

    fuel_inj_pressure[0] = ts.FuelInjectorManifold.p
    ox_inj_pressure[0] = ts.OxInjectorManifold.p

    fuel_inj_mdot[0] = ts.FuelInjector.mdot
    ox_inj_mdot[0] = ts.OxInjector.mdot

    Pc_filt = ts.MainChamber.p                # filtered Pc state for optional measurement filtering
    chamber_pressure[0] = ts.MainChamber.p
    mixture_ratio[0] = ts.MainChamber.MR
    tca_mdot[0] = ts.TCA.mdot
    thrust[0] = ts.TCA.F

    fuel_throttle_cmd[0] = ts.FuelThrottleValve.CdA
    ox_throttle_cmd[0] = ts.OxThrottleValve.CdA
    fuel_throttle_unsat[0] = ts.FuelThrottleValve.CdA
    ox_throttle_unsat[0] = ts.OxThrottleValve.CdA

    Pc_error_hist[0] = Pc_target_schedule[0] - ts.MainChamber.p
    Pc_integral_hist[0] = pid_alpha.integral
    Pc_dmeas_hist[0] = 0.0


    alpha_ff_hist[0] = np.interp(Pc_target_schedule[0], Pc_vec, alpha_vec)  # initial feedforward throttle level
    delta_alpha_hist[0] = 0.0
    alpha_cmd_hist[0] = np.clip(alpha_ff_hist[0], 0.0, 1.0)


    # ============================================================
    # Closed-Loop Simulation
    # ============================================================
    for i, t in enumerate(timespan[:-1]):

        Pc_raw = ts.MainChamber.p

        if tau_pc > 0.0:
            Pc_filt = low_pass_filter(Pc_filt, Pc_raw, dt, tau_pc)  # optional sensor filtering
        else:
            Pc_filt = Pc_raw

        Pc_measured = Pc_filt
        Pc_target = Pc_target_schedule[i]

        alpha_ff = np.interp(Pc_target, Pc_vec, alpha_vec)   # steady-state throttle level for target Pc, no feed forward rate limiting

        # Keep alpha_cmd = alpha_ff + delta_alpha inside [0, 1].
        pid_alpha.u_min = -alpha_ff
        pid_alpha.u_max = 1.0 - alpha_ff

        delta_alpha, err_pc, integ_pc, dmeas_pc = pid_alpha.update(
            target=Pc_target,
            measurement=Pc_measured,
            dt=dt,
        )

        alpha_cmd = alpha_ff + delta_alpha
        alpha_cmd = np.clip(alpha_cmd, 0.0, 1.0)    # this line is purely for redundancy

        fuel_trim = np.interp(alpha_cmd, alpha_vec, fuel_CdA_vec)  # map commanded alpha to coordinated fuel valve trim
        ox_trim = np.interp(alpha_cmd, alpha_vec, ox_CdA_vec)      # map commanded alpha to coordinated ox valve trim

        fuel_throttle_unsat[i + 1] = fuel_trim
        ox_throttle_unsat[i + 1] = ox_trim

        fuel_cmd = fuel_actuator.update(fuel_trim, dt)             # actuator enforces valve bounds + rate limit
        ox_cmd = ox_actuator.update(ox_trim, dt)

        ts.FuelThrottleValve.CdA = fuel_cmd
        ts.OxThrottleValve.CdA = ox_cmd

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

        Pc_error_hist[i + 1] = err_pc
        Pc_integral_hist[i + 1] = integ_pc
        Pc_dmeas_hist[i + 1] = dmeas_pc

        alpha_ff_hist[i + 1] = alpha_ff
        delta_alpha_hist[i + 1] = delta_alpha
        alpha_cmd_hist[i + 1] = alpha_cmd


    if print_final_condition:
        print(ts)
    
    final_condition_ts = ts

    return initial_condition_ts, final_condition_ts

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
        "alpha_ff": "#FFD700",
        "delta_alpha": "#FF69B4",
        "alpha_cmd": "#7DF9FF",
        "fuel_unsat": "#FFFFFF",
        "fuel_actual": "#AAAAAA",
        "ox_unsat": "#FF3131",
        "ox_actual": "#FFA500",
        "pc_target": "#FFD700",
    }

    fuel_sched_color = "#FFFFFF"
    ox_sched_color = "#FFA500"

    fuel_throttle_cm2 = fuel_throttle_cmd * 1e4
    ox_throttle_cm2 = ox_throttle_cmd * 1e4
    fuel_throttle_unsat_cm2 = fuel_throttle_unsat * 1e4
    ox_throttle_unsat_cm2 = ox_throttle_unsat * 1e4

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
    fig_control, axs_control = plt.subplots(3, 1, figsize=(10,10), sharex=True)

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
        alpha_ff_hist,
        color=control_colors["alpha_ff"],
        linewidth=2,
        label="alpha_ff",
    )
    axs_control[2].plot(
        timespan,
        delta_alpha_hist,
        color=control_colors["delta_alpha"],
        linewidth=2,
        label="delta_alpha",
    )
    axs_control[2].plot(
        timespan,
        alpha_cmd_hist,
        color=control_colors["alpha_cmd"],
        linewidth=2,
        label="alpha_cmd",
    )
    axs_control[2].set_ylabel("Alpha")
    axs_control[2].set_xlabel("Time (s)")
    axs_control[2].set_title("Alpha Command States")
    axs_control[2].grid(alpha=0.3)
    axs_control[2].legend()

    #fig_control.tight_layout()


    # -------------------------
    # Figure 2: Throttle commands
    # -------------------------
    fig_throttle, axs_throttle = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axs_throttle[0].plot(
        timespan,
        fuel_throttle_cm2,
        color=fuel_sched_color,
        linewidth=2,
        label="Fuel throttle CdA",
    )
    axs_throttle[0].plot(
        timespan,
        fuel_throttle_unsat_cm2,
        color=control_colors["fuel_actual"],
        linestyle="--",
        linewidth=1.75,
        label="Fuel commanded trim",
    )
    axs_throttle[0].set_ylabel("Fuel CdA (cm$^2$)")
    axs_throttle[0].set_title("Fuel Throttle Command")
    axs_throttle[0].grid(alpha=0.3)
    axs_throttle[0].legend()

    axs_throttle[1].plot(
        timespan,
        ox_throttle_cm2,
        color=control_colors["ox_actual"],
        linewidth=2,
        label="Ox throttle CdA",
    )
    axs_throttle[1].plot(
        timespan,
        ox_throttle_unsat_cm2,
        color=control_colors["ox_unsat"],
        linestyle="--",
        linewidth=1.75,
        label="Ox commanded trim",
    )
    axs_throttle[1].set_ylabel("Ox CdA (cm$^2$)")
    axs_throttle[1].set_xlabel("Time (s)")
    axs_throttle[1].set_title("Oxidizer Throttle Command")
    axs_throttle[1].grid(alpha=0.3)
    axs_throttle[1].legend()

    fig_throttle.tight_layout()


    # -------------------------
    # Figure 3: Fuel side
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
    # Figure 4: Ox side
    # -------------------------
    fig_ox, axs_ox = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axs_ox[0].plot(timespan, ox_sys_mdot, color=chamber_colors["mr"], linewidth=2)
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
    axs_ox[2].set_xlabel("Time (s)")
    axs_ox[2].set_title("Oxidizer Injector Mass Flow")
    axs_ox[2].grid(alpha=0.3)
    add_ox_throttle_overlay(axs_ox[2])

    fig_ox.tight_layout()


    # -------------------------
    # Figure 5: Chamber / nozzle
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
        color=control_colors["ox_actual"],
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

    # plt.show()


if __name__ == "__main__":
    ts = run_DualThrottle()