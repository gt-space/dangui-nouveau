import os
import sys

os.environ["QT_API"] = "PyQt6"

from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt6 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


class PlotWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas.axes.plot([0,1,2,3,4], [10,1,20,3,40]) ## fake data that im scared to touch

        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        toolbar = NavigationToolbar(self.canvas, self)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)

        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def updatePlot(self, times, data, title):
        if "(" in title:
            splitno = title.rfind("(")
            yunit = title[splitno+1:-1]
            title = title[:splitno]
            if yunit == "psi":
                yunit = "Pressure (psi)"
            elif yunit == "K":
                yunit = "Temperature (K)"
            elif yunit == "V":
                yunit = "Voltage (V)"
            elif yunit == "A":
                yunit = "Amperage (A)"
            elif yunit == "lbf":
                yunit = "Force (lbf)"
        else:
            yunit = title

        self.canvas.axes.cla()  # Clear the canvas.
        self.canvas.axes.plot(times, data)
        self.canvas.axes.set_xlabel("Time (s)")
        self.canvas.axes.set_ylabel(yunit)
        self.canvas.axes.set_title(title)
        # Trigger the canvas to update and redraw.
        self.canvas.draw()
