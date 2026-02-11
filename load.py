import os
import numpy as np
import pandas as pd
from gui import Ui_MainWindow
from plots import PlotWindow
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton

times = 0
headers = 0
data = 0
filename = "<filename>"
lastclickedplot = "<plot>"

# Reads input .csv and converts it to useful .txt
class LoadData(QMainWindow, Ui_MainWindow):
    def selectFile(self):
        global filename
        file_filter = 'Data File (*.csv)'
        fullfilepath = QFileDialog.getOpenFileName(
            parent=self,
            caption='Unga Bunga',
            directory=os.getcwd(),
            filter=file_filter
        )
        filename = str(fullfilepath)
        splitno = filename.rfind("/")
        filename = filename[splitno+1:-23]
        self.filenameText.setText(filename)

    def loadData(self):
        global times, headers, data, filename
        if os.path.exists(str(filename)):
            df = pd.read_csv(filename)
            times = df['timestamp']
            times = times - times[0]

            data = df.drop(['timestamp'], axis=1)
            data = data.reindex(sorted(data.columns), axis=1) ## future: list by type of sensor

            headers = list(data)

            self.tZero.setValue(0)
            self.tStart.setValue(0)
            self.tEnd.setValue(0)

            # Make scroll area the list of sensors
            for i in range(len(headers)-1):
                self.sensorList.addItem(headers[i+1])
        else:
            self.filenameText.setText("File does not exist!")

    def setData(self):
        global times, data
        tzero = self.tZero.value()
        tstart = self.tStart.value()
        tend = self.tEnd.value()

        if tstart <= times.iloc[0] or tstart >= times.iloc[-1]:
            tstart = times.iloc[0]
        if tend <= tstart or tend >= times.iloc[-1]:
            tend = times.iloc[-1]

        data = data[times.between(tstart, tend)]
        times = times[times.between(tstart, tend)]
        times = times - tzero

        data = data.reset_index(drop=True)
        print(data[lastclickedplot])
        times = times.reset_index(drop=True)
        print(times)

        self.tZero.setValue(tzero - tzero)
        self.tStart.setValue(tstart - tzero)
        self.tEnd.setValue(tend - tzero)

        if lastclickedplot != "<plot>":
            self.centralPlot.clear()
            self.centralPlot.plot(times, data[lastclickedplot])
    
    def plotSensors(self, clickedItem):
        global lastclickedplot 
        lastclickedplot = clickedItem.text()
        print(times)
        self.centralPlot.clear()
        self.centralPlot.plot(times, data[clickedItem.text()])

    def matplotSensors(self, clickedItem):
        return times, data[clickedItem.text()], clickedItem.text()
