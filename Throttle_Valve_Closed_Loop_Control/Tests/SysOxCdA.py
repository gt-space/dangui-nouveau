import numpy as np
import matplotlib.pyplot as plt
from TestModules import set_winplot_dark, solve_pressure_ladder, PA_PER_PSI, N_PER_LBF

# Inputs
P_tank_f  = 450.0 * PA_PER_PSI   # Pa
P_tank_ox = 400.0 * PA_PER_PSI   # Pa

sys_CdA_f  = 0.4  * 1e-4   # m^2
sys_CdA_ox_cm2   = np.linspace(0.1, 1.5, 40)
sys_CdA_ox_sweep = sys_CdA_ox_cm2 * 1e-4   # m^2

inj_CdA_f  = 0.5  * 1e-4   # m^2
inj_CdA_ox = 1.0  * 1e-4   # m^2

throat_area  = 6.05 / 1550.0 # m^2
expansion_ratio = 4.73

rho_fuel = 804.0   # kg/m^3
rho_ox   = 1104.0  # kg/m^3

cstar_eff = 0.75
cf_eff = 1.0

ambient_pressure = 14.67 * PA_PER_PSI   # Pa




# Sweep Ox CdA
x_cm2 = sys_CdA_ox_cm2  # x-axis for plots (cm^2)

P_inj_f_list = []
P_inj_ox_list = []
Pc_list = []

mdot_f_list = []
mdot_ox_list = []
mdot_tot_list = []
mr_list = []

stiff_f_list = []
stiff_ox_list = []

thrust_lbf_list = []
cstar_list = []

for sys_CdA_ox in sys_CdA_ox_sweep:
    try:
        (P_inj_f_Pa, P_inj_ox_Pa, Pc_Pa,
         mdot_f, mdot_ox, mr,
         stiff_f, stiff_ox,
         thrust_N,
         cstar) = solve_pressure_ladder(
            P_tank_f, P_tank_ox,
            sys_CdA_f, sys_CdA_ox,
            inj_CdA_f, inj_CdA_ox,
            rho_fuel, rho_ox,
            throat_area, expansion_ratio,
            cstar_eff, cf_eff,
            ambient_pressure
        )

        # Convert pressures back to psia for plotting
        P_inj_f_list.append(P_inj_f_Pa / PA_PER_PSI)
        P_inj_ox_list.append(P_inj_ox_Pa / PA_PER_PSI)
        Pc_list.append(Pc_Pa / PA_PER_PSI)

        mdot_f_list.append(mdot_f)
        mdot_ox_list.append(mdot_ox)
        mdot_tot_list.append(mdot_f + mdot_ox)

        mr_list.append(mr)

        stiff_f_list.append(stiff_f)
        stiff_ox_list.append(stiff_ox)

        # Convert thrust back to lbf for plotting
        thrust_lbf_list.append(thrust_N / N_PER_LBF)

        cstar_list.append(cstar)

    except RuntimeError:
        P_inj_f_list.append(np.nan)
        P_inj_ox_list.append(np.nan)
        Pc_list.append(np.nan)

        mdot_f_list.append(np.nan)
        mdot_ox_list.append(np.nan)
        mdot_tot_list.append(np.nan)

        mr_list.append(np.nan)

        stiff_f_list.append(np.nan)
        stiff_ox_list.append(np.nan)

        thrust_lbf_list.append(np.nan)
        cstar_list.append(np.nan)

# Plotting
set_winplot_dark()

fig, axs = plt.subplots(
    3, 2,
    figsize=(16.5, 13.5),
    sharex=False,
    gridspec_kw={"hspace": 0.38, "wspace": 0.22}
)

fig.suptitle(
    "Feed System vs Sys Ox CdA\n"
    f"Throat Area (in²): {throat_area * 1550:.2f}   |   "
    f"System CdAs (cm²) -> Fuel: {sys_CdA_f * 1e4:.1f}  ||   "
    f"Injector CdAs (cm²) -> Fuel: {inj_CdA_f * 1e4:.1f}, Ox: {inj_CdA_ox * 1e4:.1f}\n"
    f"Tank Pressures (psia) -> Fuel: {P_tank_f / PA_PER_PSI:.1f}, Ox: {P_tank_ox / PA_PER_PSI:.1f}",
    fontsize=15,
    y=0.985
)

# Global cosmetics (x-label everywhere)
for ax in axs.ravel():
    ax.grid(True)
    ax.tick_params(labelsize=11, length=4.5, width=1.1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
    ax.margins(x=0.02)
    ax.set_xlabel("Sys Ox CdA (cm²)", fontsize=11)

# Pressures
ax = axs[0, 0]
ax.plot(x_cm2, P_inj_f_list, label="Inj. Fuel", linewidth=2.8, color="red")
ax.plot(x_cm2, P_inj_ox_list, label="Inj. Ox", linewidth=2.8, color="lime")
ax.plot(x_cm2, Pc_list,       label="Chamber", linewidth=2.8, color="magenta")
ax.set_ylabel("Pressure (psia)", fontsize=11)
ax.legend(loc="best", fontsize=11)

# Mass Flows
ax = axs[0, 1]
ax.plot(x_cm2, mdot_f_list,   label="Fuel",  linewidth=2.8, color="red")
ax.plot(x_cm2, mdot_ox_list,  label="Ox",    linewidth=2.8, color="lime")
ax.plot(x_cm2, mdot_tot_list, label="Total", linewidth=2.8, color="deepskyblue")
ax.set_ylabel("Mass Flow (kg/s)", fontsize=11)
ax.legend(loc="best", fontsize=11)

# Mixture Ratios
ax = axs[1, 0]
ax.plot(x_cm2, mr_list, linewidth=2.8, color="deepskyblue")
ax.set_ylabel("Mixture Ratio (O/F)", fontsize=11)
ax.legend(loc="best", fontsize=11)

# Injector Stiffnesses
ax = axs[1, 1]
ax.plot(x_cm2, stiff_f_list,  label="Fuel", linewidth=2.8, color="red")
ax.plot(x_cm2, stiff_ox_list, label="Ox",   linewidth=2.8, color="lime")
ax.set_ylabel("Stiffness (% of Pc)", fontsize=11)
ax.legend(loc="best", fontsize=11)

# Thrust
ax = axs[2, 0]
ax.plot(
    x_cm2, thrust_lbf_list,
    label=f"Cf eff = {cf_eff*100:.1f}%\nExp. Ratio = {expansion_ratio:.1f}\nAmbient: {ambient_pressure/PA_PER_PSI:.1f} psia",
    linewidth=2.8,
    color="magenta"
)
ax.set_ylabel("Thrust (lbf)", fontsize=11)
ax.legend(loc="best", fontsize=11)

# C*
ax = axs[2, 1]
ax.plot(
    x_cm2, cstar_list,
    label=f"c* eff = {cstar_eff*100:.1f}%",
    linewidth=2.8,
    color="yellow"
)
ax.set_ylabel("c* (m/s)", fontsize=11)
ax.legend(loc="best", fontsize=11)

# Consistent x-limits
xmin, xmax = float(np.nanmin(x_cm2)), float(np.nanmax(x_cm2))
for ax in axs.ravel():
    ax.set_xlim(xmin, xmax)

plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
plt.show()