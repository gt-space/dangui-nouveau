## this is the main file. this calls the GUI

import sys
import numpy as np
import pandas as pd

from gui import Ui_MainWindow
from load import LoadData
from plots import PlotWindow

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

    # All buttons
    def selectButtonPushed(self):
        LoadData.selectFile(self)

    def loadButtonPushed(self):
        LoadData.loadData(self)

    def setButtonPushed(self):
        LoadData.setData(self)

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

# Executing the app
app = QApplication(sys.argv)

window = MainWindow()
window.show()
app.exec()
