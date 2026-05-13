import os
import numpy as np
import pandas as pd
from gui import Ui_MainWindow
from plots import PlotWindow
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton

class Database(QMainWindow, Ui_MainWindow):
    def saveData(self):
        print("here")
