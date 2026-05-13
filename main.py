## this is the main file. this calls the GUI

import sys
import numpy as np
import pandas as pd

from gui import Ui_MainWindow
from load import LoadData
from plots import PlotWindow
from calcs import DataCalcs

from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton
from PyQt6 import QtCore
import pyqtgraph as pg

#test 
dataIsSet = False
dataIsGet = False # not sure if i need to globalize these tbh
dataIsLoad = False
dataIsSelect = False

QMainWindow, Ui_MainWindow = pg.Qt.loadUiType("DANGUI-GUI/form.ui")

class MainWindow(QMainWindow, Ui_MainWindow):
    # Custom init/setup stuff
    def __init__(self, *args, obj=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.selectDataButton.clicked.connect(self.selectButtonPushed)
        self.loadDataButton.clicked.connect(self.loadButtonPushed)
        self.setDataButton.clicked.connect(self.setButtonPushed)
        self.sensorList.itemClicked.connect(self.sensorListPushed)
        self.sensorList.itemDoubleClicked.connect(self.sensorListDoubleClicked)
        self.fuelbuttongroup.buttonClicked.connect(self.fuelbuttongroupClicked)
        self.oxbuttongroup.buttonClicked.connect(self.oxbuttongroupClicked)

        self.solveEqnsButton.clicked.connect(self.solveEquations)
        self.setPropertiesButton.clicked.connect(self.setProperties)
        self.loxDensity.textChanged.connect(self.updateDensity)
        self.fuelDensity.textChanged.connect(self.updateDensity)

        self.headers = None
        self.data = None

    # All buttons
    def selectButtonPushed(self):
        global dataIsSelect, dataIsLoad, dataIsSet, dataIsGet
        dataIsLoad = False
        LoadData.selectFile(self)
        if self.filenameText.toPlainText() != '':
            dataIsSelect = True
        else:
            print("choose a goddamn file!")

    def loadButtonPushed(self):
        global dataIsSelect, dataIsLoad, dataIsSet, dataIsGet
        dataIsSet = False
        dataIsGet = False
        self.headers = LoadData.loadData(self, dataIsSelect) # loadData returns headers
        dataIsLoad = True

    def setButtonPushed(self):
        global dataIsSelect, dataIsLoad, dataIsSet, dataIsGet
        if dataIsLoad == False:
            if dataIsSelect == True:
                print("You have not loaded your most recently selected data. Click the Load Data button.")
                return
            else:
                print("Select Data first.")
                return
        data = LoadData.setData(self) # setData returns data without timestamps
        self.data = data.to_numpy() # converst pd frame to np array
        dataIsSet = True

    def sensorListPushed(self, clickedItem):
        LoadData.plotSensors(self, clickedItem)

    def sensorListDoubleClicked(self, clickedItem):
        times, data, title = LoadData.matplotSensors(self, clickedItem)
        self.showPlotWindow(times, data, title)

    def fuelbuttongroupClicked(self, clickedItem):
        if clickedItem.text()=="Jet-A":
            self.fuelDensity.setPlainText("800")
        elif clickedItem.text()=="Water":
            self.fuelDensity.setPlainText("1000")
        elif clickedItem.text()=="Ethanol":
            self.fuelDensity.setPlainText("789")
        else:
            self.fuelDensity.setPlainText("0")

    def oxbuttongroupClicked(self, clickedItem):
        if clickedItem.text()=="LOX":
            self.loxDensity.setPlainText("1100")
        elif clickedItem.text()=="LN2":
            self.loxDensity.setPlainText("807")
        elif clickedItem.text()=="NOX":
            self.loxDensity.setPlainText("1220")
        else:
            self.loxDensity.setPlainText("0")

    # Plot window setup
    def showPlotWindow(self, times, data, title):
        self.w = PlotWindow()
        self.w.show()
        PlotWindow.updatePlot(self.w, times, data, title)
    
    def setProperties(self):
        global dataIsSelect, dataIsLoad, dataIsSet, dataIsGet
        if dataIsSelect==False:
            print("Select Data first.")
            return
        if dataIsLoad==False:
            print("Please load data.")
            return
        if dataIsSet==False:
            print('ay bro you gotta set the data in the Main tab. press the button that says Set Data. get tf outta here bruh')
            return
        DataCalcs.getData(self, dataIsSet)
        dataIsGet = True

    def solveEquations(self):
        global dataIsSet, dataIsGet
        DataCalcs.solver(self, dataIsSet, dataIsGet)

    def updateDensity(self):
        DataCalcs.updateDensity(self)        

    def findData(self, header): # helper function to return data of specified header
                                # inherited by calcs.py
                                # what to do when does not work? suggest all zeroes and inside error message, don't stop calcs
        try:
            idx = self.headers.index(header)
        except ValueError:
            idx = -1
        if type(self.data[0,idx]) != 'str':
            foundData = self.data[:,idx:idx+1].astype(float)
        else:
            foundData = self.data[:,idx:idx+1]

        return foundData


# Executing the app
app = QApplication(sys.argv)

window = MainWindow()
window.show()
app.exec()
