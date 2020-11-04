from tkinter import filedialog
import pandas as pd
import sys
import pickle
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from PyQt5 import QtCore, QtGui, QtWidgets
from sklearn import preprocessing #Para preprocesar los datos
from sklearn.model_selection import train_test_split #seleccionar toma de datos para entrenamiento y pruebas
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score #matriz de Confusión
from datetime import date
from datetime import datetime
from pathlib import Path
import ctypes  # An included library with Python install. 

from PyQt5 import QtCore, QtGui, QtWidgets

class Clasificar(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(50, 50, 281, 121))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.getModelo)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(450, 50, 301, 121))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.getCSV)
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(50, 200, 701, 211))
        self.textEdit.setObjectName("textEdit")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 420, 281, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_2.setGeometry(QtCore.QRect(280, 420, 401, 41))
        self.textEdit_2.setObjectName("textEdit_2")
        self.textEdit_2.textChanged.connect(self.comprobarBoton)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(110, 480, 491, 71))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.clasificacion)
        self.pushButton_3.setEnabled(False)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Importar Modelo"))
        self.pushButton_2.setText(_translate("MainWindow", "Importar Dataset Test"))
        self.label.setText(_translate("MainWindow", "Nombre archivo prediccion:"))
        self.pushButton_3.setText(_translate("MainWindow", "Guardar Prediccion"))

    #Funcion para comprobar que se ha escrito el nombre del archivo .csv a generar con la prediccion
    def comprobarBoton(self):
        if(self.textEdit_2.toPlainText()!=''):
            self.pushButton_3.setEnabled(True)
        else:
            self.pushButton_3.setEnabled(False)

    #Funcion para seleccionar el modelo para realizar las pruebas
    def getModelo (self):
        filepath = filedialog.askopenfilename()
        print(filepath)
        global archivoModelo
        archivoModelo = filepath
        #Pop-up para informar al usuario que se ha seleccionado con exito el modelo
        ctypes.windll.user32.MessageBoxW(0, "Modelo importado con exito", "Mensaje", 0)
        return archivoModelo
    
    #Funcion para importar un dataset y transformarlo en un dataframe
    def getCSV (self):
        filepath = filedialog.askopenfilename()
        print(filepath)
        df = pd.read_csv(filepath, delimiter =";", encoding = "ISO-8859-1")
        global archivoCSV
        archivoCSV = df
        ctypes.windll.user32.MessageBoxW(0, "CSV de test importado con exito", "Mensaje", 0)
        return archivoCSV
    
    #Funcion para guardar el dataset con la prediccion
    def guardarDatasetPrediccion(self, dataframe):
        root = Path(".") #Cogemos el directorio actual
        fileNameCSV = self.textEdit_2.toPlainText()+".csv" #Seleccionamos el fichero con el nombre del textEdit
        path = root / fileNameCSV #Seleccionamos la ruta
        dataframe.to_csv(path)  #Gurdamos el nuevo csv en la ruta  
        
    #Funcion para hacer el clasificador
    def clasificacion(self):
        datosFicheroPruebas = archivoCSV #Variable global que contiene los datos
        modelo = archivoModelo #Variable global que contiene el modelo
        with open(modelo, 'rb') as pickle_file:
            model = pickle.load(pickle_file) 

        datosParaClasificar = datosFicheroPruebas.copy() #Creamos una copia del df para no modificar el original
        keysTestModel = list(datosParaClasificar.head()) #Cabeceras de los datos
        columnas = len(keysTestModel)
        #print(columnas)
        #print(keysTestModel)
        print(datosParaClasificar)
        le = preprocessing.LabelEncoder() #Metodo capaz de tranformar strings en valores para trabajar con ellos

        for c in range(columnas): #Recorremos las cabeceras
            if (datosParaClasificar[keysTestModel[c]].dtype == object): #Comprobamos si los datos de la cabecera c son strings
                datosParaClasificar[keysTestModel[c]] = le.fit_transform(datosParaClasificar[keysTestModel[c]]) #Transformamos la informacion de la cabecera en valores 
                datosParaClasificar[keysTestModel[c]] = datosParaClasificar[keysTestModel[c]].astype('category') #Transformamos el tipo de valor de los datos a categorico
        
        x = datosParaClasificar.values
        print(x)
        # Utilizamos la prediccion para los datos x
        print(model.predict(x))
        
        dataTest_exportado = datosFicheroPruebas.copy() #Creamos otra copia del df original para guardarlo
        dataTest_exportado.insert(11, "Prediccion", model.predict(x), True) #Incorporamos nuevo atributo PREDICCION
        self.guardarDatasetPrediccion(dataTest_exportado)
        self.textEdit.append(str(dataTest_exportado))
        ctypes.windll.user32.MessageBoxW(0, "CSV con la prediccion generado con exito", "Mensaje", 0)
    


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Clasificar()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

