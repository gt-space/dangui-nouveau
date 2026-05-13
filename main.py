## this is the main file. this calls the GUI

import sys
import numpy as np
import pandas as pd

from gui import Ui_MainWindow
from load import LoadData
from plots import PlotWindow
from calcs import DataCalcs
from controlLoop import HETSModel

from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton
from PyQt6 import QtCore
import pyqtgraph as pg


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

        self.solveEqnsButton.clicked.connect(self.solveEquations)
        self.setPropertiesButton.clicked.connect(self.setProperties)
        self.loxDensity.textChanged.connect(self.updateDensity)
        self.fuelDensity.textChanged.connect(self.updateDensity)

        self.headers = None
        self.data = None

        self.dualThrottleButton.clicked.connect(self.HETSDualThrottle)

    # All buttons
    def selectButtonPushed(self):
        LoadData.selectFile(self)

    def loadButtonPushed(self):
        self.headers = LoadData.loadData(self) # loadData returns headers

    def setButtonPushed(self):
        data = LoadData.setData(self) # setData returns data without timestamps
        self.data = data.to_numpy() # converst pd frame to np array

    def sensorListPushed(self, clickedItem):
        LoadData.plotSensors(self, clickedItem)

    def sensorListDoubleClicked(self, clickedItem):
        times, data, title = LoadData.matplotSensors(self, clickedItem)
        self.showPlotWindow(times, data, title)

    # Plot window setup
    def showPlotWindow(self, times, data, title):
        self.w = PlotWindow()
        self.w.show()
        PlotWindow.updatePlot(self.w, times, data, title)
    
    def setProperties(self):
        DataCalcs.getData(self)

    def solveEquations(self):
        DataCalcs.solver(self)

    def updateDensity(self):
        DataCalcs.updateDensity(self)        

    def findData(self, header): # helper function to return data of specified header
                                # inherited by calcs.py
        idx = self.headers.index(header)
        if type(self.data[0,idx]) != 'str':
            foundData = self.data[:,idx:idx+1].astype(float)
        else:
            foundData = self.data[:,idx:idx+1]

        return foundData
    

    # FLUID MODEL

    def HETSDualThrottle(self):
        HETSModel.dualThrottle(self) 



# Executing the app
app = QApplication(sys.argv)

window = MainWindow()
window.show()
app.exec()
