from tkinter import filedialog
import pandas as pd
import sys
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from PyQt5 import QtCore, QtGui, QtWidgets
from sklearn import preprocessing #Para preprocesar los datos
from sklearn.model_selection import train_test_split #seleccionar toma de datos para entrenamiento y pruebas
from sklearn.metrics import confusion_matrix, classification_report #matriz de Confusión
from datetime import date
from datetime import datetime
from pathlib import Path
global archivoCSV
import ctypes  # An included library with Python install.   



#from clasificar import Clasificar


class Entrenar(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(270, 380, 221, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(170, 380, 151, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(240, 40, 301, 51))
        self.pushButton.clicked.connect(self.getCSV)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_2.setGeometry(QtCore.QRect(20, 110, 751, 251))
        self.textEdit_2.setObjectName("listView")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(80, 420, 161, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(250, 430, 291, 31))
        self.textEdit.setObjectName("textEdit")
        self.textEdit.textChanged.connect(self.comprobarBoton)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(250, 482, 261, 51))
        self.pushButton_2.clicked.connect(self.entrenamiento)
        self.pushButton_2.setEnabled(False)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
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
        self.comboBox.setItemText(0, _translate("MainWindow", "Naive Bayes"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Neighbors Classifier"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Decision Tree"))
        self.label.setText(_translate("MainWindow", "Algoritmo:"))
        self.pushButton.setText(_translate("MainWindow", "Importar Dataset Entrenamiento"))
        self.label_2.setText(_translate("MainWindow", "Nombre del modelo:"))
        self.pushButton_2.setText(_translate("MainWindow", "Guardar Modelo"))
        
    #Funcion para comprobar que se ha escrito el nombre del modelo a generar
    def comprobarBoton(self):
        if(self.textEdit.toPlainText()!=''):
            self.pushButton_2.setEnabled(True)
        else:
            self.pushButton_2.setEnabled(False)

    #Funcion para importar un dataset y transformarlo en un dataframe
    def getCSV (self):
        filepath = filedialog.askopenfilename()
        print(filepath)
        df = pd.read_csv(filepath, delimiter =";", encoding = "ISO-8859-1")
        global archivoCSV
        archivoCSV = df
        print(archivoCSV)
        ctypes.windll.user32.MessageBoxW(0, "CSV de entrenamiento importado con exito", "Mensaje", 0)
        #caracteristica s= list(df.head())
        #print(caracteristicas)
        #print(df)
        return archivoCSV
    
    #Funcion para seleccionar el algoritmo de aprendizaje automatico
    def seleccionAlgoritmoAprendizaje(self): 
        seleccion = self.comboBox.currentText()
        if seleccion == "Naive Bayes":
            modelo = GaussianNB()
            print("Gauss")
        if seleccion == "Neighbors Classifier":
            modelo = KNeighborsClassifier()
            print("KN")
        if seleccion == "Decision Tree":
            modelo = tree.DecisionTreeClassifier()
            print("DT")

        return modelo
    
    #Funcion para guardar el modelo generado
    def guardarModelo(self,clasificador):
        root = Path(".") #Directorio actual
        ficheroSAV = self.textEdit.toPlainText()+".sav"
        path = root / ficheroSAV   #Directorio carpeta y nombre del archivo
        fichero = open(path, 'wb')  #Abrimos el fichero para la escritura
        fichero = pickle.dump(clasificador, fichero) #Guardamos el modelo en el fichero
        ctypes.windll.user32.MessageBoxW(0, "Modelo generado con exito", "Mensaje", 0)
    
    #Funcion de entrenamiento del modelo
    def entrenamiento(self): 
        datosFicheroEntrenamiento = archivoCSV
        print("archivoCSV:", archivoCSV)

        label = datosFicheroEntrenamiento['Retraso'] #Marcamos como target la caracteristica Retraso
        datosFicheroEntrenamiento = datosFicheroEntrenamiento.drop(['Retraso'], axis=1) #Eliminamos la columna Retraso
        keysTrainModel = list(datosFicheroEntrenamiento.head()) #Cabeceras del df
        columns = len(keysTrainModel)
        le = preprocessing.LabelEncoder() #Metodo capaz de tranformar strings en valores para trabajar con ellos

        for c in range(columns):
            if (datosFicheroEntrenamiento[keysTrainModel[c]].dtype == object): #Comprobamos si los datos de la cabecera c son strings
                datosFicheroEntrenamiento[keysTrainModel[c]] = le.fit_transform(datosFicheroEntrenamiento[keysTrainModel[c]]) #Transformamos la informacion de la cabecera en valores 
                datosFicheroEntrenamiento[keysTrainModel[c]] = datosFicheroEntrenamiento[keysTrainModel[c]].astype('category') #Transformamos el tipo de valor de los datos a categorico
        
        x = datosFicheroEntrenamiento.values #Valores con los que vamos a trabajar
        y = label.values #Valores del target
        #Dividimos las variables de los valores entre train y test con un 20% durante el entrenamiento
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
        clasificador = self.seleccionAlgoritmoAprendizaje()
        clasificador.fit(x_train, y_train) #Utilizamos los datos train de las variables para crear el modelo
        y_pred = clasificador.predict(x_test) #Prediccion del target en funcion a los datos de los demas atributos
        report =  classification_report(y_test, y_pred, output_dict=True) #Calcula precision y recall
        now = datetime.now()
        #Agregamos al textEdit para mostrar en la interfaz
        self.textEdit_2.append("Fecha y hora de creacion del modelo: ")#añadimos la informacion al textEdit 
        self.textEdit_2.append(str(now))
        self.textEdit_2.append("\n")
        self.textEdit_2.append("Matriz de confusion: ")
        self.textEdit_2.append(str(confusion_matrix(y_test, y_pred)))
        self.textEdit_2.append("\n")
        self.textEdit_2.append("Report del algoritmo de aprendizaje automatico: ")
        self.textEdit_2.append(str(report))
        self.textEdit_2.append("\n")
        self.textEdit_2.append("Dataframe del CSV de entrenamiento: ")
        self.textEdit_2.append(str(datosFicheroEntrenamiento))
        #self.textEdit_2.setText(str(now))
        #self.textEdit_2.setText("\n")
        #self.textEdit_2.setText("Matriz de confusion: ")
        #self.textEdit_2.setText(str(confusion_matrix(y_test, y_pred)))
        #self.textEdit_2.setText("\n")
        #self.textEdit_2.setText("Report del algoritmo de aprendizaje: ")
        #self.textEdit_2.setText(str(report))
        #self.textEdit_2.setText("\n")
        #self.textEdit_2.setText("Prediccion: ")
        #self.textEdit_2.setText(str(datosFicheroEntrenamiento))
        #print("Prediccion ",y_pred)
        #print("Report ",report)
        #print("Matriz ",confusion_matrix(y_test, y_pred))
        return self.guardarModelo(clasificador) #llamamos a funcion para que guarde el modelo con el clasificador entrenado
        
    
#today = date.today()
#now = datetime.now()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Entrenar()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

