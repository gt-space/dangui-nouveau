"""
Fluid name alias mapping for CoolProp compatibility.

All keys must be lowercase.
"""

FLUID_NAME_BANK = {
    # Water
    "water": "Water",
    "h2o": "Water",

    # Oxygen / LOX
    "oxygen": "Oxygen",
    "o2": "Oxygen",
    "lox": "Oxygen",
    "liquid oxygen": "Oxygen",

    # Hydrogen
    "hydrogen": "Hydrogen",
    "h2": "Hydrogen",
    "lh2": "Hydrogen",
    "liquid hydrogen": "Hydrogen",

    # Nitrogen
    "nitrogen": "Nitrogen",
    "n2": "Nitrogen",
    "ln2": "Nitrogen",
    "liquid nitrogen": "Nitrogen",

    # Methane
    "methane": "Methane",
    "ch4": "Methane",
    "lng": "Methane",

    # Ethanol
    "ethanol": "Ethanol",
    "c2h5oh": "Ethanol",
    "alcohol": "Ethanol",

    # Isopropanol
    "isopropanol": "Isopropanol",
    "ipa": "Isopropanol",
    "isopropyl alcohol": "Isopropanol",

    # Kerosene / RP-1 surrogate
    "rp-1": "n-Dodecane",
    "rp1": "n-Dodecane",
    "rocket propellant-1": "n-Dodecane",
    "kerosene": "n-Dodecane",
    "kerosense": "n-Dodecane",
    "jet-a": "n-Dodecane",
    "jeta": "n-Dodecane",
    "n-dodecane": "n-Dodecane",
    "dodecane": "n-Dodecane",

    # Propane / Butane
    "propane": "Propane",
    "c3h8": "Propane",
    "butane": "n-Butane",
    "n-butane": "n-Butane",

    # Helium
    "helium": "Helium",
    "he": "Helium",
}