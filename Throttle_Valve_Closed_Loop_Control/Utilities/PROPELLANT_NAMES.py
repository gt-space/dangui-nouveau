"""
Propellant name alias mapping for RocketCEA compatibility.

All keys must be lowercase.
All values must be valid RocketCEA propellant names.
"""

PROPELLANT_NAME_BANK = {
    # -------------------------
    # OXIDIZERS
    # -------------------------
    "lox": "LOX",
    "liquid oxygen": "LOX",
    "oxygen": "LOX",
    "o2": "LOX",
    "ox": "LOX",

    "n2o": "N2O",
    "nitrous": "N2O",
    "nitrous oxide": "N2O",

    "h2o2": "H2O2",
    "peroxide": "H2O2",
    "hydrogen peroxide": "H2O2",

    "f2": "F2",
    "fluorine": "F2",

    # -------------------------
    # FUELS
    # -------------------------

    # RP-1 / Kerosene family
    "rp-1": "RP-1",
    "rp1": "RP-1",
    "rocket propellant-1": "RP-1",
    "kerosene": "RP-1",
    "jet-a": "RP-1",
    "jeta": "RP-1",
    "n-dodecane": "RP-1",
    "dodecane": "RP-1",
    "ndodecane": "RP-1",

    # Hydrogen
    "lh2": "LH2",
    "liquid hydrogen": "LH2",
    "hydrogen": "LH2",
    "h2": "LH2",

    # Methane
    "methane": "CH4",
    "ch4": "CH4",
    "lng": "CH4",

    # Ethanol
    "ethanol": "C2H5OH",
    "alcohol": "C2H5OH",
    "c2h5oh": "C2H5OH",

    # IPA
    "isopropanol": "C3H8O",
    "ipa": "C3H8O",

    # Hydrazine family
    "hydrazine": "N2H4",
    "n2h4": "N2H4",

    "mmh": "MMH",
    "monomethylhydrazine": "MMH",

    "udmh": "UDMH",
    "unsymmetrical dimethylhydrazine": "UDMH",

    # Ammonia
    "nh3": "NH3",
    "ammonia": "NH3",
}