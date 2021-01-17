""" Vamos a jugar Gato: Este es un proyecto de Machine Learnin donde en base a un juego de gato nuestro algoritmo sabra si ganamos o perdemos
en un juego de gato, en este caso sera un juego de un solo jugador solo para realizar la prueba, si vemos nuestra data es un problema de tipo de
clasificacion, la clasificacion es que si pertenecemos al grupo de ganar o perder
Data a utilizar: https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/ """

#Vamos a comenzar importando los modulos necesarios
#Importamos Pandas para importar y modificar nuestra Data
import pandas as pd

#Importamos NumPy para uso de arreglos
import numpy as np

#Importamos el separador de datos en datos de prueba y entrenamiento
from sklearn.model_selection import train_test_split

#Importamos nuestro algoritmo de clasificacion
from sklearn.ensemble import RandomForestClassifier

#Comenzamos con importar la data
DatosGato = pd.read_csv('ProyectosML/GatoJuego/Data/tic-tac-toe.csv')

#Verificamos el contenido de nuestros datos
print(DatosGato.head()) #Contenido de la Data
print(DatosGato.info()) #Informacion del tipo de datos
print(DatosGato.isnull().sum()) #Verificar si hay datos nulos

""" Primero vamos a agregar las columnas:
- Superior(Sup)
- Inferior(Inf)
- Centro(Cent) """
DatosGato.columns = ['Izq-Sup', 'Cent-Sup', 'Der-Sup', 'Izq-Cent', 'Centro', 'Der-Cent', 'Izq-Infe', 'Cent-Infe', 'Der-Infe', 'Resultado']
#Vemos si se realizaron los cambios
print(DatosGato.head())

#Viendo el contenido de nuestro datos vamos a realizar el cambio de objeto a numerico
#Comenzamos con las columnas que contiene los datos 'x, o, b'
for i in range(9):
    Columna = DatosGato.columns[i]
    DatosGato[Columna].replace(('x', 'o', 'b'), (1, 2, 3), inplace=True)
#Para ahorrar tiempo realizaremos un ciclo for, porque no encontre otra opcion mejor

#Ahora vamos con la columna de resultado, donde si gana x es '1' y si pierde x es '0'
DatosGato['Resultado'].replace(('positive', 'negative'), (1, 0), inplace=True)

#Verificamos si los datos son de tipo numerico
print(DatosGato.info())
print(DatosGato.head())

#Vamos con el algoritmo, comenzamos con separar los datos del resultado y solamente el resultado para asi entrenar a nuestro algoritmo
X = np.array(DatosGato.drop(['Resultado'], 1))
Y = np.array(DatosGato['Resultado'])

#Vamos a usar el separador de datos de prueba y datos de entrenamiento
X_Entre, X_Prueba, Y_Entre, Y_Prueba = train_test_split(X, Y)#Usaremos toda la data

#Seleccionamos nuestro algoritmo
Algoritmo = RandomForestClassifier()

#Entrenamos el algoritmo
Algoritmo.fit(X_Entre, Y_Entre)

#Vemos el score
print(Algoritmo.score(X_Prueba, Y_Prueba))

#Tenemos un buen resultado asi que vamos a probarlo y realizaremos una prediccion
print(Algoritmo.predict([[2, 2, 1, 2, 1, 2, 1, 2 ,2]]))#Nos tiene que arrojar un 1
print(Algoritmo.predict([[1, 1, 2, 1, 2, 1, 2, 1, 1]]))#Nos tiene que arrojar un 0