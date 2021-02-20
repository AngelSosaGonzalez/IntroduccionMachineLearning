"""Evaluacion de carro: Este proyecto de evaluacion de autos veremos si el coche que vamos a comprar es una buena compra o no, como problema de
clasificacion vamos a utilizar algoritmos de esta rama.
- Data: https://archive.ics.uci.edu/ml/machine-learning-databases/car/ """

#Primero vamos a importar los modulos necesarios
#Importamos pandas para importar la Data y para modificar los datos
import pandas as pd

#Importamos NumPy para el manejo de arreglos
import numpy as np

#Importamos el separador de datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split

#Importamos nuestro algoritmo
from sklearn.ensemble import RandomForestClassifier

#Importamos nuestra Data
DatosCarro = pd.read_csv('Machine Learning/ProyectosML/CarroEvaluacion/Data/car.csv')

#Vemos el contenido de la data
print(DatosCarro.head())#Contenido de la Data
print(DatosCarro.info())#Informacion de la Data
print(DatosCarro.isnull().sum())#Datos vacios

#Comenzamos con el preprocesamiento
#Agregamos columnas
DatosCarro.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'desicion']
#Cambiamos los datos Object a tipo numerico
DatosCarro['buying'].replace(('vhigh', 'high', 'med', 'low'), (4, 3, 2, 1), inplace=True)
DatosCarro['maint'].replace(('vhigh', 'high', 'med', 'low'), (4, 3, 2, 1), inplace=True)
DatosCarro['doors'].replace(('2', '3', '4', '5more'), (1, 2, 3, 4), inplace=True)
DatosCarro['persons'].replace(('2', '4', 'more'), (1, 2, 3),inplace=True)
DatosCarro['lug_boot'].replace(('small', 'med', 'big'), (1, 2, 3), inplace=True)
DatosCarro['safety'].replace(('low', 'med', 'high'), (1, 2, 3), inplace=True)
DatosCarro['desicion'].replace(('unacc', 'acc', 'good', 'vgood'), (1, 2, 3, 4), inplace=True)

#Ahora vamos a urilizar nuestro algoritmo
#Separamos los datos con la columna final
X = np.array(DatosCarro.drop(['desicion'], 1))
Y = np.array(DatosCarro['desicion'])

#Separamos los datos de entrenamiento y prueba
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y)

#Imbocamos nuestro algortimo
Algortimo = RandomForestClassifier()

#Entrenamos nuestro algoritmo
Algortimo.fit(X_Train, Y_Train)

#Vemos el score
print(Algortimo.score(X_Test, Y_Test))
#Dio un buen puntaje vamos a realizar predicciones

print(Algortimo.predict([[1, 2, 3, 2, 2, 2]]))#Primera prediccion
print(Algortimo.predict([[1, 2, 1, 2, 2, 2]]))#Segunda prediccion