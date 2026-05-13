'''

INSTRUCTIONS TO RUN THIS FILE:

Becuase this file is not in the root directory and has relative imports
running it is not as easy just hitting run.

In the command terminal for the root directory (the overall project folder), run:

---> python3 -m Tests.test_steady

(replace 'python3' with just 'python' if you're not using Python 3)

'''


import numpy as np
import matplotlib.pyplot as plt
from Utilities import set_winplot_dark, solve_system_CdAs, get_density, get_pressure
from Utilities.Constants import PA_PER_PSI, M2_PER_IN2, N_PER_LBF
from Network.Components import *
from Network import TestStand, Balance
from scipy.optimize import root

set_winplot_dark()


# -----------------------------
# Build baseline test stand
# -----------------------------
FuelTank = Tank("Fuel Tank", "jet-a", 550 * PA_PER_PSI, 70/1000, temperature=300)
OxTank   = Tank("Oxidizer Tank", "LOX", 500 * PA_PER_PSI, 70/1000, temperature=90)

FuelRunline = Line("Fuel Runline")
OxRunline   = Line("Oxidizer Runline")

FuelThrottle = Valve("Fuel Throttle Valve", 0.5e-4)
OxThrottle   = Valve("Oxidizer Throttle Valve", 1e-4)

FuelInjectorManifold = InjectorManifold("Fuel Manifold", 350 * PA_PER_PSI, temperature=500)
OxInjectorManifold   = InjectorManifold("Oxidizer Manifold", 350 * PA_PER_PSI)

FuelInjector = Orifice("Fuel Injector", 0.5e-4)
OxInjector   = Orifice("Oxidizer Injector", 1e-4)

Chamber = CombustionChamber("Main Chamber", 300 * PA_PER_PSI)
TCA     = Nozzle("Nozzle", 6.05 * M2_PER_IN2)

Atmosphere = Drain("Ambient")

HETS = TestStand(
    "HETS",
    FuelTank, OxTank,
    FuelRunline, OxRunline,
    FuelThrottle, OxThrottle,
    FuelInjectorManifold, OxInjectorManifold,
    FuelInjector, OxInjector,
    Chamber, TCA, Atmosphere
)

#solved = HETS.steady_state()
#print(solved.__str__(units="US"))


'''
# -----------------------------
# Sweep conditions
# -----------------------------
MR_target = 3.0

Pc_psia = np.linspace(150, 350, 10)
Pc = Pc_psia * PA_PER_PSI

CdA_f_list = []
CdA_ox_list = []

for Pc_i in Pc:

    try:
        CdA_fuel_sol, CdA_ox_sol, _ = solve_system_CdAs(
            HETS,
            Pc_target = Pc_i,
            MR_target = MR_target,
            verbose = False
        )

        CdA_f_list.append(CdA_fuel_sol)
        CdA_ox_list.append(CdA_ox_sol)

    except Exception:
        CdA_f_list.append(np.nan)
        CdA_ox_list.append(np.nan)


CdA_f = np.array(CdA_f_list)
CdA_ox = np.array(CdA_ox_list)


# -----------------------------
# Convert units
# -----------------------------
CdA_f_in2 = CdA_f / M2_PER_IN2
CdA_ox_in2 = CdA_ox / M2_PER_IN2


# -----------------------------
# Plot
# -----------------------------

plt.figure(figsize=(8,5))

plt.plot(Pc_psia, CdA_f_in2, linewidth=2, label="Fuel System CdA")
plt.plot(Pc_psia, CdA_ox_in2, linewidth=2, label="Ox System CdA")

plt.xlabel("Chamber Pressure (psia)")
plt.ylabel("System CdA (in²)")

plt.title(f"Required System CdA vs Chamber Pressure\nMR = {MR_target}")

plt.grid(True)
plt.legend()

plt.tight_layout()
#plt.show()
'''

'''
# -----------------------------
# Sweep throat area and solve
# -----------------------------
At_in2 = np.linspace(4.0, 6.5, 31)          # 31 points from 5 to 6.5 in^2
At_m2  = At_in2 * M2_PER_IN2

Pc_Pa  = np.full_like(At_m2, np.nan, dtype=float)
F_N    = np.full_like(At_m2, np.nan, dtype=float)

# Use previous solution as the next initial guess for faster/more robust convergence
x0 = None

for i, At in enumerate(At_m2):
    try:
        # set new throat area
        HETS.TCA.At = float(At)

        # solve; pass previous x0 if your steady_state accepts it
        solved = HETS.steady_state(x0=x0)

        # record results
        Pc_Pa[i] = solved.MainChamber.p
        F_N[i]   = solved.TCA.F

        # update x0 for continuation (P_inj_f, P_inj_ox, Pc)
        x0 = [solved.FuelInjectorManifold.p, solved.OxInjectorManifold.p, solved.MainChamber.p]

        # (optional) update HETS baseline to the solved state for continuation behavior
        HETS = solved

    except Exception as e:
        print(f"[WARN] At = {At_in2[i]:.3f} in^2 failed: {e}")

# Convert Pc to psi for plotting
Pc_psi = Pc_Pa / PA_PER_PSI


set_winplot_dark()

plt.figure()
plt.plot(At_in2, Pc_psi, linewidth=2, color="red")
plt.xlabel("Throat Area $A_t$ [in$^2$]")
plt.ylabel("Chamber Pressure $P_c$ [psi]")
plt.title("Steady-State Chamber Pressure vs Throat Area")
plt.grid(True, alpha=0.6)
plt.tight_layout()

plt.figure()
plt.plot(At_in2, F_N / N_PER_LBF, linewidth=2, color='yellow')
plt.xlabel("Throat Area $A_t$ [in$^2$]")
plt.ylabel("Thrust $F$ [lbf]")
plt.title("Steady-State Thrust vs Throat Area")
plt.grid(True, alpha=0.6)
plt.tight_layout()

#plt.show()
'''






Pc_balance = Balance(
    tune="TCA.At",
    measure="MainChamber.p",
    target=300 * PA_PER_PSI,
    bounds=(4 * M2_PER_IN2, 8 * M2_PER_IN2),
    tol=2000.0,   # Pa (~0.3 psi)
    name="Balance At until Pc = 300 psia"
)

solved = HETS.steady_state_with_balance(Pc_balance)


F_balance = Balance(
    tune="TCA.At",
    measure="TCA.F",
    target=2500.0 * N_PER_LBF,      # N
    bounds=(5 * M2_PER_IN2, 6.5 * M2_PER_IN2),
    tol=1.0,            # 1 Newton tolerance
)

#solved = HETS.steady_state_with_balance(F_balance)


MR_balance = Balance(
    tune="OxThrottleValve.CdA",
    measure="MainChamber.MR",
    target=2,
    bounds=(1e-6, 1e-4),
    tol=1e-5,   #
)
solved = HETS.steady_state_with_balance(MR_balance)


FuelStiffness20 = Balance(
    tune="TCA.At",
    measure="FuelInjector.stiffness",
    target=0.2,                      
    bounds=(3e-5, 2e-4),              # pick a reasonable At bracket for your engine
    tol=0.05,                         # percent tolerance (0.05% ~ very tight)
    name="Tune throat area until fuel injector stiffness = 20%",
)

#solved = HETS.steady_state_with_balance(FuelStiffness20)


mdot_5 = Balance(
    tune="FuelThrottleValve.CdA",
    measure="TCA.mdot",
    target=5.0,          # kg/s
    bounds=(1e-6, 2e-4),
    tol=0.01,            # kg/s
    name="Tune fuel throttle until total mdot = 5 kg/s",
)

#solved = HETS.steady_state_with_balance(mdot_5)


FuelStiffness20 = Balance(
    tune="TCA.At",
    measure=lambda s: 100.0 *
        (s.FuelInjectorManifold.p - s.MainChamber.p)
        / s.MainChamber.p,
    target=20.0,
    bounds=(3e-5, 2e-4),
    tol=0.01,
    name="Tune throat until fuel injector stiffness = 20%",
)

#solved = HETS.steady_state_with_balance(FuelStiffness20)

print(solved.__str__(units="US"))





