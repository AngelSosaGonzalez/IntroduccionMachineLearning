""" Poker: Realizaremos una aplicacion de Machine Learning utilizando el modulo de Sklearn, el objetivo de este proyecto es ver saber que tipo de
juego sacamos dependiendo de la mano de poker que tenemos nos dara que tipo de juego tenemos, nuestra Data es para algoritmos de clasificacion 
por lo que vamos a probar con diferentes algoritmos.
Link de la Data: https://archive.ics.uci.edu/ml/machine-learning-databases/poker/ """

#Importamos los modulos necesarios
#Pandas para importar nuestra data y preprocesamiento
import pandas as pd

#Importamos Numpy para los arreglos de entrenamiento
import numpy as np

#Importamos el separador de los datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split

#Importamos nuestro algoritmo
from sklearn.ensemble import RandomForestClassifier

#Comezamos con importar nuestras Datas
#Data de entrenamiento
DatosPokerTrain = pd.read_csv('ProyectosML/CartasPoker/Data/poker-hand-training-true.csv')
print(DatosPokerTrain.head())
#Data de prueba
DatosPokerTest = pd.read_csv('ProyectosML/CartasPoker/Data/poker-hand-testing.csv')
print(DatosPokerTest.head())

""" Le daremos columnas a nuestras Datas:
F: Figura (Hearts, Spades, Diamonds, Clubs)
C: Carta (1-13)
Mano: Es el resultado de la mano que tenemos """
DatosPokerTrain.columns = ['F1', 'C1', 'F2', 'C2', 'F3', 'C3', 'F4', 'C4', 'F5', 'C5', 'Mano']
print(DatosPokerTrain.head())
DatosPokerTest.columns = ['F1', 'C1', 'F2', 'C2', 'F3', 'C3', 'F4', 'C4', 'F5', 'C5']
print(DatosPokerTest.head())

#Vemos si la info de los datos y si hay datos perdidos
#Info
print(DatosPokerTrain.info())
print(DatosPokerTest.info())
#Datos perdidos
print(DatosPokerTrain.isnull().sum())
print(DatosPokerTest.isnull().sum())
#No hay datos perdidos asi que no haremos preprocesamiento

#Vamos a separar los datos de la columna de Mano para entrenar los datos
#Data sin la columna da Mano
X = np.array(DatosPokerTrain.drop(['Mano'], 1))
#Solamente le columna de Mano
Y = np.array(DatosPokerTrain['Mano'])

#Separamos nuestros datos en entrenamiento y prueba
X_Entre, X_Prueba, Y_Entre, Y_Prueba = train_test_split(X, Y, test_size = 0.2)

#Teniendo estos datos definiremos nuestro algoritmo
Algoritmo = RandomForestClassifier(max_depth=15)
#Su puntaje es muy bajo para realizar una prediccion pero es el algortmo con mas puntaje

#Vamos a entrenar nuestro algoritmo
Algoritmo.fit(X_Entre, Y_Entre)

#Prediccion y score
Prediccion = Algoritmo.predict(DatosPokerTest)
print(Algoritmo.score(X_Prueba, Y_Prueba))

#Realizaremos predicciones
PrediccionAlgoritmo = pd.DataFrame({'Juego': Prediccion})
print(PrediccionAlgoritmo.tail())