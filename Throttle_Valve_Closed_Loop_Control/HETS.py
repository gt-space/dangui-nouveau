from Utilities.Constants import PA_PER_PSI, M2_PER_IN2
from Network.Components import *
from Network import TestStand

filename = "alpha_map.parquet"

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
