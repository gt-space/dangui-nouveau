# runs the dualThrottle control loop and access all the results 
# future: access the graphs and display on click 


'''import subprocess

subprocess.run(
    ["python", "DualThrottle.py"],
    cwd="Throttle-Valve-Closed-Loop-Control"
)s
print('done')'''

import sys, os
import numpy as np

from gui import Ui_MainWindow
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton

ctrl_dir = os.path.join(os.path.dirname(__file__), "Throttle_Valve_Closed_Loop_Control")
if ctrl_dir not in sys.path:
    sys.path.insert(0, ctrl_dir)
    
from Throttle_Valve_Closed_Loop_Control import DualThrottle
from Throttle_Valve_Closed_Loop_Control.Utilities.Constants import PA_PER_PSI, LBM_PER_KG, LBF_PER_N, IN2_PER_M2

class HETSModel(QMainWindow, Ui_MainWindow):
    def dualThrottle(self):
        initial_ts, final_ts = DualThrottle.run_DualThrottle()

        # STEADY STATE - initial_ts
        # make it so this calcs first
        init = {
            # Pressures / Temperatures / Densities 
            "fuel_tank_p_pa":    str(round(initial_ts.FuelTank.p, 2)),
            "fuel_tank_p_psi":   str(round(initial_ts.FuelTank.p / PA_PER_PSI, 2)),
            "ox_tank_p_pa":      str(round(initial_ts.OxTank.p, 2)),
            "ox_tank_p_psi":     str(round(initial_ts.OxTank.p / PA_PER_PSI, 2)),
            "fuel_inj_p_pa":     str(round(initial_ts.FuelInjectorManifold.p, 2)),
            "fuel_inj_p_psi":    str(round(initial_ts.FuelInjectorManifold.p / PA_PER_PSI, 2)),
            "ox_inj_p_pa":       str(round(initial_ts.OxInjectorManifold.p, 2)),
            "ox_inj_p_psi":      str(round(initial_ts.OxInjectorManifold.p / PA_PER_PSI, 2)),
            "chamber_p_pa":      str(round(initial_ts.MainChamber.p, 2)),
            "chamber_p_psi":     str(round(initial_ts.MainChamber.p / PA_PER_PSI, 2)),

            # Mass Flow 
            "fuel_mdot_kgs":         str(round(initial_ts.FuelInjector.mdot, 4)),
            "fuel_mdot_lbms":        str(round(initial_ts.FuelInjector.mdot * LBM_PER_KG, 4)),
            "ox_mdot_kgs":           str(round(initial_ts.OxInjector.mdot, 4)),
            "ox_mdot_lbms":          str(round(initial_ts.OxInjector.mdot * LBM_PER_KG, 4)),
            "total_mdot_kgs":        str(round(initial_ts.FuelInjector.mdot + initial_ts.OxInjector.mdot, 4)),
            "total_mdot_lbms":       str(round((initial_ts.FuelInjector.mdot + initial_ts.OxInjector.mdot) * LBM_PER_KG, 4)),
            "MR":                    str(round(initial_ts.MainChamber.MR, 4)),
            "fuel_inj_stiffness":    str(round(initial_ts.FuelInjector.stiffness * 100, 2)),
            "ox_inj_stiffness":      str(round(initial_ts.OxInjector.stiffness * 100, 2)),

            # Nozzle / Performance
            "thrust_n":              str(round(initial_ts.TCA.F, 2)),
            "thrust_lbf":            str(round(initial_ts.TCA.F * LBF_PER_N, 2)),

            # CdA Summary
            "fuel_throttle_m2":      str(round(initial_ts.FuelThrottleValve.CdA, 3)),
            "fuel_throttle_in2":     str(round(initial_ts.FuelThrottleValve.CdA * IN2_PER_M2, 4)),
            "ox_throttle_m2":        str(round(initial_ts.OxThrottleValve.CdA, 3)),
            "ox_throttle_in2":       str(round(initial_ts.OxThrottleValve.CdA * IN2_PER_M2, 4)),
            "fuel_inj_m2":           str(round(initial_ts.FuelInjector.CdA, 3)),
            "fuel_inj_in2":          str(round(initial_ts.FuelInjector.CdA * IN2_PER_M2, 4)),
            "ox_inj_m2":             str(round(initial_ts.OxInjector.CdA, 3)),
            "ox_inj_in2":            str(round(initial_ts.OxInjector.CdA * IN2_PER_M2, 4)),
        }

        # Display on GUI
        self.fuel_tank_p_pa.setText(init["fuel_tank_p_pa"])
        self.fuel_tank_p_psi.setText(init["fuel_tank_p_psi"])
        self.ox_tank_p_pa.setText(init["ox_tank_p_pa"])
        self.ox_tank_p_psi.setText(init["ox_tank_p_psi"])
        self.fuel_inj_p_pa.setText(init["fuel_inj_p_pa"])
        self.fuel_inj_p_psi.setText(init["fuel_inj_p_psi"])
        self.ox_inj_p_pa.setText(init["ox_inj_p_pa"])
        self.ox_inj_p_psi.setText(init["ox_inj_p_psi"])
        self.chamber_p_pa.setText(init["chamber_p_pa"])
        self.chamber_p_psi.setText(init["chamber_p_psi"])
        # ambient pa
        # ambient psi

        self.fuel_mdot_kgs.setText(init["fuel_mdot_kgs"])
        self.fuel_mdot_lbms.setText(init["fuel_mdot_lbms"])
        self.ox_mdot_kgs.setText(init["ox_mdot_kgs"])
        self.ox_mdot_lbms.setText(init["ox_mdot_lbms"])
        self.total_mdot_kgs.setText(init["total_mdot_kgs"])
        self.total_mdot_lbms.setText(init["total_mdot_lbms"])
        self.MR.setText(init["MR"])
        self.fuel_inj_stiffness.setText(init["fuel_inj_stiffness"])
        self.ox_inj_stiffness.setText(init["ox_inj_stiffness"])

        self.thrust_n.setText(init["thrust_n"])
        self.thrust_lbf.setText(init["thrust_lbf"])

        self.fuel_throttle_m2.setText(init["fuel_throttle_m2"])
        self.fuel_throttle_in2.setText(init["fuel_throttle_in2"])
        self.ox_throttle_m2.setText(init["ox_throttle_m2"])
        self.ox_throttle_in2.setText(init["ox_throttle_in2"])
        self.fuel_inj_m2.setText(init["fuel_inj_m2"])
        self.fuel_inj_in2.setText(init["fuel_inj_in2"])
        self.ox_inj_m2.setText(init["ox_inj_m2"])
        self.ox_inj_in2.setText(init["ox_inj_in2"])
        

# need max throttle condition within steady state info
# need a copy of this for final state



# CHANGED ALPHA MAP FILE PATH in DualThrottle, 114
# changed wildcard import, 77 (wildcard not allowed in function)
# added 715, 716 
# return statements (and setting intial and final conditions below their respective print statements), 397, 395, 164