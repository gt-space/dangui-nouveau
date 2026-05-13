import numpy as np
import matplotlib.pyplot as plt
from TestModules import (
    set_winplot_dark,
    PA_PER_PSI,
    solve_SysCdAs,
    get_mdot,
)

# Inputs
Pc_psia = np.linspace(200.0, 350.0, 60)
Pc_pa   = Pc_psia * PA_PER_PSI
MR = 2.5

P_tank_f  = 500.0 * PA_PER_PSI   # Pa
P_tank_ox = 450.0 * PA_PER_PSI   # Pa

inj_CdA_f  = 0.5  * 1e-4   # m^2
inj_CdA_ox = 1.0  * 1e-4   # m^2

throat_area = 6.05 / 1550.0      # m^2
expansion_ratio = 4.73 

rho_fuel = 804.0                 # kg/m^3
rho_ox   = 1104.0                # kg/m^3

cstar_eff = 1.0
cf_eff = 1.0  

ambient_pressure = 14.67 * PA_PER_PSI   # Pa 

# Sweep Pc
P_inj_f_psia = []
P_inj_ox_psia = []
stiff_f = []
stiff_ox = []

mdot_f = []
mdot_ox = []
mdot_tot = []

sysCdA_f_cm2 = []
sysCdA_ox_cm2 = []

for Pc in Pc_pa:
    try:
        # --- System CdAs + injector pressures ---
        sysCdA_f, sysCdA_o, P_inj_f, P_inj_o = solve_SysCdAs(
            Pc, MR,
            P_tank_f, P_tank_ox,
            inj_CdA_f, inj_CdA_ox,
            rho_fuel, rho_ox,
            throat_area, cstar_eff
        )

        # System CdAs (cm^2)
        sysCdA_f_cm2.append(sysCdA_f * 1e4)
        sysCdA_ox_cm2.append(sysCdA_o * 1e4)

        # Injector pressures (psia)
        P_inj_f_psia.append(P_inj_f / PA_PER_PSI)
        P_inj_ox_psia.append(P_inj_o / PA_PER_PSI)

        # Injector stiffness (% of Pc)
        Pc_psia_i = Pc / PA_PER_PSI
        stiff_f.append((P_inj_f / PA_PER_PSI - Pc_psia_i) / Pc_psia_i * 100.0)
        stiff_ox.append((P_inj_o / PA_PER_PSI - Pc_psia_i) / Pc_psia_i * 100.0)

        # Mass flows (kg/s)
        mdot_total = get_mdot(Pc, MR, throat_area, cstar_eff)
        mdot_fuel  = mdot_total / (1.0 + MR)
        mdot_oxid  = mdot_total - mdot_fuel

        mdot_f.append(mdot_fuel)
        mdot_ox.append(mdot_oxid)
        mdot_tot.append(mdot_total)

    except RuntimeError:
        sysCdA_f_cm2.append(np.nan)
        sysCdA_ox_cm2.append(np.nan)

        P_inj_f_psia.append(np.nan)
        P_inj_ox_psia.append(np.nan)

        stiff_f.append(np.nan)
        stiff_ox.append(np.nan)

        mdot_f.append(np.nan)
        mdot_ox.append(np.nan)
        mdot_tot.append(np.nan)

# Convert to arrays
P_inj_f_psia  = np.array(P_inj_f_psia)
P_inj_ox_psia = np.array(P_inj_ox_psia)
stiff_f       = np.array(stiff_f)
stiff_ox      = np.array(stiff_ox)

mdot_f   = np.array(mdot_f)
mdot_ox  = np.array(mdot_ox)
mdot_tot = np.array(mdot_tot)

sysCdA_f_cm2  = np.array(sysCdA_f_cm2)
sysCdA_ox_cm2 = np.array(sysCdA_ox_cm2)

# Plot
set_winplot_dark()

fig, axs = plt.subplots(
    2, 2,
    figsize=(14.5, 10.5),
    gridspec_kw={"wspace": 0.28, "hspace": 0.34}
)

fig.suptitle(
    "Constant Mixture Ratio Sweep vs Chamber Pressure\n"
    f"MR = {MR:.2f}   |   Throat Area = {throat_area*1550:.2f} in²   |   "
    f"Tanks (psia): Fuel {P_tank_f/PA_PER_PSI:.1f}, Ox {P_tank_ox/PA_PER_PSI:.1f}   |   "
    f"Injector CdAs (cm²): Fuel {inj_CdA_f*1e4:.2f}, Ox {inj_CdA_ox*1e4:.2f}\n"
    f"c* eff = {cstar_eff*100:.1f}%   |   "
    f"Exp. Ratio = {expansion_ratio:.2f}   |   Ambient = {ambient_pressure/PA_PER_PSI:.2f} psia",
    fontsize=14,
    y=0.97
)

# Global cosmetics
for ax in axs.ravel():
    ax.grid(True)
    ax.tick_params(labelsize=11, length=4.5, width=1.1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
    ax.margins(x=0.02)
    ax.set_xlabel("Pc (psia)", fontsize=11)

# Injector pressures
ax = axs[0, 0]
ax.plot(Pc_psia, P_inj_f_psia, label="Inj. Fuel", linewidth=2.8, color="red")
ax.plot(Pc_psia, P_inj_ox_psia, label="Inj. Ox",   linewidth=2.8, color="lime")
ax.set_ylabel("Injector Pressure (psia)", fontsize=11)
ax.legend(loc="best", fontsize=11)

# Injector stiffness
ax = axs[0, 1]
ax.plot(Pc_psia, stiff_f,  label="Fuel Stiffness", linewidth=2.8, color="red")
ax.plot(Pc_psia, stiff_ox, label="Ox Stiffness",   linewidth=2.8, color="lime")
ax.set_ylabel("Injector Stiffness (% of Pc)", fontsize=11)
ax.legend(loc="best", fontsize=11)

# Mass flows
ax = axs[1, 0]
ax.plot(Pc_psia, mdot_f,   label="Fuel",  linewidth=2.8, color="red")
ax.plot(Pc_psia, mdot_ox,  label="Ox",    linewidth=2.8, color="lime")
ax.plot(Pc_psia, mdot_tot, label="Total", linewidth=2.8, color="deepskyblue")
ax.set_ylabel("Mass Flow (kg/s)", fontsize=11)
ax.legend(loc="best", fontsize=11)

# System CdAs
ax = axs[1, 1]
ax.plot(Pc_psia, sysCdA_f_cm2,  label="Fuel", linewidth=2.8, color="red")
ax.plot(Pc_psia, sysCdA_ox_cm2, label="Ox",   linewidth=2.8, color="lime")
ax.set_ylabel("System CdA (cm²)", fontsize=11)
ax.legend(loc="best", fontsize=11)

# Consistent x-limits
xmin, xmax = float(np.nanmin(Pc_psia)), float(np.nanmax(Pc_psia))
for ax in axs.ravel():
    ax.set_xlim(xmin, xmax)

plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
plt.show()