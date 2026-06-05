import numpy as np
import matplotlib.pyplot as plt

from gui import Ui_MainWindow
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton
from load import plotCustom

class DataCalcs(QMainWindow, Ui_MainWindow):

    def updateDensity(self):
        self.loxRhoBox.setPlaceholderText(self.loxDensity.toPlainText())
        self.fuelRhoBox.setPlaceholderText(self.fuelDensity.toPlainText())

    def getData(self, dataIsSet):
        
        self.loxRhoBox.setPlaceholderText(self.loxDensity.toPlainText())
        self.fuelRhoBox.setPlaceholderText(self.fuelDensity.toPlainText())

        # to handle custom equation density based on whether user inputted data or just used main page density
        if self.loxRhoBox.text() == '':
            self.loxRhoCustom = float(self.loxRhoBox.placeholderText())
        else:
            self.loxRhoCustom = float(self.loxRhoBox.text()) # custom equation lox density, for some reason i made this a QLineEdit instead ¯\_(ツ)_/¯
        if self.fuelRhoBox.text() == '':
            self.fuelRhoCustom = float(self.loxRhoBox.placeholderText())
        else:
            self.fuelRhoCustom = float(self.fuelRhoBox.text()) # custom equation fuel density 

        # getting values for automatic equations 
            # sys and inj mdots
        self.loxp1 = self.findData('OIPT(psi)') # 'loxp1' contains a header, loxp1 is a numpy array of corresponding values
        self.loxp2 = self.findData('CHPT1(psi)') 
        self.fuelp1 = self.findData('FIPT(psi)')
        self.fuelp2 = self.findData('CHPT1(psi)')
        self.ODP = self.findData('ODP(psi)')

        self.OTPT = self.findData('OTPT(psi)')
        self.FTPT = self.findData('FTPT(psi)')
        self.OIPT = self.findData('OIPT(psi)')
        self.FIPT = self.findData('FIPT(psi)')
        self.CHPT = self.findData('CHPT1(psi)')
        self.loxSysCda = float(self.loxSysBox.toPlainText())
        self.loxInjCda = float(self.loxInjBox.toPlainText())
        self.loxRho = float(self.loxDensity.toPlainText())
        self.loxVenCda = float(self.loxVenBox.toPlainText())

        self.fuelSysCda = float(self.fuelSysBox.toPlainText())
        self.fuelInjCda = float(self.fuelInjBox.toPlainText())
        self.fuelRho = float(self.fuelDensity.toPlainText())

        # getting values for custom equations
        self.loxp1Cus = self.findData(self.loxDP1Combo.currentText())
        self.loxp2Cus = self.findData(self.loxDP2Combo.currentText())

        self.fuelp1Cus = self.findData(self.fuelDP1Combo.currentText())
        self.fuelp2Cus = self.findData(self.fuelDP2Combo.currentText())

        if self.loxSysCdACombo.currentText() == 'InjCdA':
            self.loxCusCda = float(self.loxInjBox.toPlainText())
        else: # SysCdA
            self.loxCusCda = float(self.loxSysBox.toPlainText())

        if self.fuelSysCdACombo.currentText() == 'InjCdA':
            self.fuelCusCda = float(self.fuelInjBox.toPlainText())
        else: # SysCdA
            self.fuelCusCda = float(self.fuelSysBox.toPlainText())  

        dataIsGet = True
        return dataIsGet

    def solver(self, dataIsSet, dataIsGet):
        if dataIsSet == False:
            print("dude ur like 2 steps too early go back to the Main tab and fucking do it again. bruh")
            return
        if dataIsGet == False:
            print("ay bro press the mf Set Properties button bruh")
            return

        inlet_ID = 0.844 # in
        throat_ID = 0.4375 # in
        inlet_ID2 = np.power(inlet_ID / 39.37 / 2, 2) # pre-processing inletID value
        throat_ID2 = np.power(throat_ID / 39.37 / 2, 2) # pre-processing throatID value

        # calculating Mdots
        self.loxSysMdot = self.loxSysCda/100**2 * np.sqrt(2* self.loxRho * np.abs(self.loxp1-self.loxp2)*6895) #
        self.loxInjMdot = self.loxInjCda/100**2 * np.sqrt(2* self.loxRho * np.abs(self.loxp1-self.loxp2)*6895) #
        self.loxVenMdot = self.loxVenCda*np.pi*inlet_ID2 * np.sqrt(2*self.loxRho* np.abs(self.ODP)*6895 / (np.power(inlet_ID2 / (np.pi*throat_ID2), 2) -1)) # kg/s, Bernoulli's Principle
        
        self.fuelSysMdot = self.fuelSysCda/100**2 * np.sqrt(2* self.fuelRho * np.abs(self.fuelp1-self.fuelp2)*6895) #
        self.fuelInjMdot = self.fuelInjCda/100**2 * np.sqrt(2* self.fuelRho * np.abs(self.fuelp1-self.fuelp2)*6895) #
        self.fuelVenMdot = self.findData('FLOW(kg/s)') # just draw values from coriolis data when it exists. may need to concern ourselves with venturi when Darcy

        # calculating and setting average values
        # TODO figure out how the averaging works for MATLAB
        self.loxMdotSysAvg.setText(str(round(np.average(self.loxSysMdot),3)))
        self.loxMdotInjAvg.setText(str(round(np.average(self.loxInjMdot),3)))
        self.loxMdotVenAvg.setText(str(round(np.average(self.loxVenMdot),3)))
        
        self.fuelMdotSysAvg.setText(str(round(np.average(self.fuelSysMdot),3)))
        self.fuelMdotInjAvg.setText(str(round(np.average(self.fuelInjMdot),3)))
        self.fuelMdotFlowAvg.setText(str(round(np.average(self.fuelVenMdot),3)))

        # calculating CdAs
        # can throw these equations into a helper function if needed
        self.loxSysCdaInj = self.loxInjMdot / (np.sqrt(2*self.loxRho * np.abs((self.OTPT-self.OIPT)+1)*6895))*100**2 # cm^2, Sys CdA using venturi Mdot and pressure drop across system
        self.loxSysCdaVen = self.loxVenMdot / (np.sqrt(2*self.loxRho * np.abs((self.OTPT-self.OIPT)+1)*6895))*100**2 # cm^2, Sys CdA using Mdot from injector and pressure drop across system 
        self.loxInjCdaSys = self.loxSysMdot / (np.sqrt(2*self.loxRho * np.abs((self.OTPT-self.OIPT)+1)*6895))*100**2
        self.loxInjCdaVen = self.loxInjMdot / (np.sqrt(2*self.loxRho * np.abs((self.OTPT-self.OIPT)+1)*6895))*100**2

        # calculating and setting average values
        self.loxSysCdaInjAvg.setText(str(round(np.average(self.loxSysCdaInj),3))) 
        self.loxSysCdaVenAvg.setText(str(round(np.average(self.loxSysCdaVen),3))) 
        self.loxInjCdaSysAvg.setText(str(round(np.average(self.loxInjCdaSys),3))) 
        self.loxInjCdaVenAvg.setText(str(round(np.average(self.loxInjCdaVen),3))) 

         # can throw these equations into a helper function if needed
        self.fuelSysCdaInj = self.fuelInjMdot / (np.sqrt(2*self.fuelRho * np.abs((self.FTPT-self.FIPT)+1)*6895))*100**2 # cm^2, Sys CdA using venturi Mdot and pressure drop across system
        self.fuelSysCdaVen = self.fuelVenMdot / (np.sqrt(2*self.fuelRho * np.abs((self.FTPT-self.FIPT)+1)*6895))*100**2 # cm^2, Sys CdA using Mdot from injector and pressure drop across system 
        self.fuelInjCdaSys = self.fuelSysMdot / (np.sqrt(2*self.fuelRho * np.abs((self.FTPT-self.FIPT)+1)*6895))*100**2
        self.fuelInjCdaVen = self.fuelInjMdot / (np.sqrt(2*self.fuelRho * np.abs((self.FTPT-self.FIPT)+1)*6895))*100**2

        # calculating and setting average values
        self.fuelSysCdaInjAvg.setText(str(round(np.average(self.fuelSysCdaInj),3))) 
        self.fuelSysCdaVenAvg.setText(str(round(np.average(self.fuelSysCdaVen),3))) 
        self.fuelInjCdaSysAvg.setText(str(round(np.average(self.fuelInjCdaSys),3))) 
        self.fuelInjCdaVenAvg.setText(str(round(np.average(self.fuelInjCdaVen),3))) 

        # set internally for future calcs
        self.loxInjMdot = self.loxInjCdaSys / 100**2 * np.sqrt(2*self.loxRho*np.abs(self.OIPT-self.CHPT)*6895)

        O_MR = 0
        F_MR = 1
        # find MR
        if self.loxCdACombo.currentText()=="LOX Sys":
            O_MR = float(self.loxMdotSysAvg.text())
        elif self.loxCdACombo.currentText()=="LOX Inj":
            O_MR = float(self.loxMdotInjAvg.text())
        elif self.loxCdACombo.currentText()=="LOX Venturi":
            O_MR = float(self.loxMdotVenAvg.text())
        if self.fuelCdACombo.currentText()=="Fuel Sys":
            F_MR = float(self.fuelMdotSysAvg.text())
        elif self.fuelCdACombo.currentText()=="Fuel Inj":
            F_MR = float(self.fuelMdotInjAvg.text())
        elif self.fuelCdACombo.currentText()=="Fuel FM":
            F_MR = float(self.fuelMdotFlowAvg.text())
        MR = O_MR/F_MR
        self.AvgMRBox.setText(str(round(MR,3)))

        # custom equations
        # note that each time you edit custom equation, you have to re-set it
        if self.loxGraphCheck.isChecked() or self.loxAppendCheck.isChecked():
            
            self.loxCusMdot = self.loxCusCda/100**2 * np.sqrt(2* self.loxRhoCustom * np.abs(self.loxp1Cus-self.loxp2Cus)*6895)
            self.fuelCusMdot = self.fuelCusCda/100**2 * np.sqrt(2* self.fuelRhoCustom * np.abs(self.fuelp1Cus-self.fuelp2Cus)*6895)

            # always append to output (see how matlab does it)
            # TODO DENSITY ISN'T CUSTOM
            # TODO if custom equation was edited and set wasn't pressed again, print error
            
            if self.loxGraphCheck.isChecked():
                plotCustom('LOX Mdot', self.loxCusMdot)
                print(self.loxDP1Combo.currentText())
                plt.show()
            if self.fuelGraphCheck.isChecked():
                plotCustom('Fuel Mdot', self.fuelCusMdot)
                plt.show()

            print('SUCCESSFUL CUSTOM CALCS')

        print("SUCCESSFUL CALCS")

        
        # np.savetxt("loxSysMdotPython.csv", loxSysMdot, delimiter=",")



