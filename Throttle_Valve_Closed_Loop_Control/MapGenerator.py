
import copy
from HETS import HETS

from Utilities.Constants import PA_PER_PSI


# --- Inputs ---
#filename = "alpha_map.parquet"
#filename = "oxCdA_map.parquet"
filename = "fuelCdA_map.parquet"
test_stand = HETS


ts = copy.deepcopy(test_stand)

'''
# --- Generate alpha map ---
pcmr_map = ts.generate_alpha_map(
    MR_target=2.0,
    Pc_min=250 * PA_PER_PSI,            # Lower bound Pc on map
    Pc_max=310 * PA_PER_PSI,            # Upper bound Pc on map
    Pc_step=0.1 * PA_PER_PSI,             # dPc on map
    fuel_CdA_range=(1.0e-6, 1.5e-4),    # steady state solver limits on Fuel side
    ox_CdA_range=(1.0e-6, 1.5e-4),      # steady state solver limits on Ox side
    return_dataframe = True,            # If you want the map as a pandas dataframe
    save_parquet = True,                # Make this True if you want a parquet file
    parquet_filename = filename,        # parquet file names
    verbose=True,                      # Set this to true if you want to see if the generation failed
)
'''

'''
# --- Generate Ox CdA map ---
ox_map = ts.generate_ox_cda_map(
    fuel_CdA=test_stand.FuelThrottleValve.CdA,
    ox_CdA_range=(2.2e-5, 1.0e-3),
    ox_CdA_step=1.0e-6,
    return_dataframe=True,
    save_parquet=True,
    parquet_filename=filename,
    verbose=True,
)
'''


'''

# --- Generate Fuel CdA map ---
fuel_map = ts.generate_fuel_cda_map(
    ox_CdA=test_stand.OxThrottleValve.CdA,
    fuel_CdA_range=(1.5e-5, 1.0e-3),
    fuel_CdA_step=1.0e-6,
    return_dataframe=True,
    save_parquet=True,
    parquet_filename=filename,
    verbose=True,
)
'''

'''
# test import
import pandas as pd

df = pd.read_parquet('oxCdA_map.parquet')
print(df.head())
'''