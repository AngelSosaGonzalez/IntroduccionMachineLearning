""" Regrecion polinomial: Este proyecto aplicaremos el algoritmo de RP (regresion polinomial), para esto aplicaremos
el mismo caso que el anterior algoritmo (IntroRegresionLineal.py, esta en la misma carpeta), que son los datos en la casas de boston
usaremos el anterior Data de Boston, antes de comenzar quiero aclarar que este proyecto se basa (o copia mas bien) del curso de 
Machine Learning del canal de: AprendeIA con Ligdi Gonzalez, fuente:https://www.youtube.com/watch?v=lnilw1y6n2o&list=PLJjOveEiVE4Dk48EI7I-67PEleEC5nxc3&index=21 """

#Comenzamos importando los modulos necesarios
#Numpy para la creacion de Arrays
import numpy

#Importamos el modulo de DataSet y el algoritmo (linear_model)
from sklearn import datasets, linear_model

#Importamos matplotlib para graficar
import matplotlib.pyplot as plt

#Importaremos el modulo de separacion de datos (Entrenamiento y Pureba)
from sklearn.model_selection import train_test_split

#Importaremos el modulo para usar Regresion polinomial
from sklearn.preprocessing import PolynomialFeatures

#Importamos nuestros datos a una variable
DatosBoston = datasets.load_boston()

#Ahora vamos a seleccionar los datos necesarios para esto usaremos la cantidad de habitaciones
NumHabit = DatosBoston.data[:, numpy.newaxis, 5]

#Obtendremos las etiquetas de los datos
Columns = DatosBoston.target

#Podremos graficar los datos de nuestro DataSet
#scatter es un diagrama de dispercion
plt.scatter(NumHabit, Columns)
plt.show()

#Vamos a separar los datos de entrenamiento y prueba
#Recordemos que test_size sera el tamaño de las muestras
X_Entrena, X_Prueba, Y_Entrena, Y_Prueba = train_test_split(NumHabit, Columns, test_size = 0.2)

#Definiremos el grado del polinomio
GradPoli = PolynomialFeatures(degree=2)

#Se transforma las caracteriticas exsitentes en caracteristicas de mayo grado
X_Entrena_Poli = GradPoli.fit_transform(X_Entrena)
X_Prueba_Poli = GradPoli.fit_transform(X_Prueba)

#Vamos a invocar a nuestro algoritmo
RegPoli = linear_model.LinearRegression()

""" Vamos a entrenar nuestro algoritmo (En esta parte en vez de utilizar los datos de entrenamiento 
usando la funcion de train_test_split, ahora usaremos la que transformamos con la funcion de 
GradPoli.fit_transform(X_Entrena)) """
RegPoli.fit(X_Entrena_Poli, Y_Entrena)

#Vamos a calcular el score de nuestro algoritmo
print(RegPoli.score(X_Prueba_Poli, Y_Prueba))
#Vemos el resultamos es muy bajo para el algoritmo

#Realizaremos una prediccion
Y_Prediccion = RegPoli.predict(X_Prueba_Poli)

#Ahora lo graficaremos para verificar cual es el problema (O solamente para visualizar el algoritmo)
#Mostramos los datos de prueba
plt.scatter(X_Prueba, Y_Prueba)
#Mostraremos el rango de obtencion de dato, esto lo mostraremos con una linea roja
plt.plot(X_Prueba, Y_Prediccion, color = 'red', linewidth = 2)
#Por ultimo mostraremos los datos
plt.show()