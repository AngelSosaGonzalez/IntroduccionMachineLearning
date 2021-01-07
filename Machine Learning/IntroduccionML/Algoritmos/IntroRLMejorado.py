""" Algoritmo de regrecion: En estadísticas, regresión lineal es una aproximación para modelar la relación entre una variable escalar dependiente “y” y una o mas variables explicativas nombradas con “X”.
En este proyecto veremos el uso de este algoritmo para Machine Learning, pero en base a la anterior en este algoritmo lo mejoraremos para sacar 
un mejor score, para este proyecto vamos a a basarnos en el curso de Machine Learning del canal
de: AMP Tech, fuente: https://www.youtube.com/watch?v=38rBECdCv3A&list=PLA050nq-BHwMr0uk7pPJUqRgKRRGhdvKb&index=3"""

#Importaremos los modulos para realzar la practica
#Importamos el modulo de KNN (Vecinos cercanos)
from sklearn.neighbors import KNeighborsRegressor
#Para esta practica usaremos el set de datos Boston (este nos lo da Sklearn)
from sklearn.datasets import load_boston
#Para dividir nuestra Data en pruebas y entrenamiento
from sklearn.model_selection import train_test_split
#importaremos el algoritmo de regresion lineal
from sklearn.linear_model import LinearRegression, Ridge

#Importamos los datos en una variable, este es otra forma de importar datos
DatosBoston = load_boston()

""" Vamos a ocupar la funcion para el entrenamiento y pruebas (train_test_split) 
para el uso de esta funcion lo dividiremos en 4 variables donde se guardara los 
valores de entrenamiento y purebas, para esto los dividiremos en X y Y (para pruebas y entrenamiento) """
X_Entrenamiento, X_Prueba, Y_Entrenamiento, Y_Prueba = train_test_split(DatosBoston.data, DatosBoston.target)

""" Vamos a darle un valor a nuestra K, esto significa cuantos vecinos vamos a tomar como referencia
para saber al momento de ingresar un nuevo registro o clasificar un registro nuevo, este toma como referencia el 
numero de vecino introducidos (osea el valor de K) al momento de darle un valor este tomara el valor 
de los vecinos que el nuevo registro tomara de referencia para asi clasificarlo """
#Como en el curso toma 7 vecino igual tomaremos 7 vecinos
K = KNeighborsRegressor(n_neighbors=4)

#Ahora vamor a entrenar nuestro clasificador, para entrenarlo usaremos los valores de X_Entrenamiento y Y_Entrenamiento, usando la funcion "fit"
K.fit(X_Entrenamiento, Y_Entrenamiento)

#Calcularemos la presicion de nuestro clasificador
print(K.score(X_Prueba, Y_Prueba))
""" Al usar KNN para regresion nos da puntajes de 40 a 50 de precision, da entender que no es un buen algoritmo
por lo que usaremos otra solucion """

#Comando basico de Python, eliminamos la variable del algoritmo
del K

#Algoritmo de regrecion lineal
#Importamos nuestro algoritmo a una variable
AlgoritmoRL =  LinearRegression()

#Entrenamos nuestro algoritmo (Usamos los mismo parametroa que el algoritmo anterior)
AlgoritmoRL.fit(X_Entrenamiento, Y_Entrenamiento)

#Ahora veamos la precision de nuestro algoritmo 
print(AlgoritmoRL.score(X_Prueba, Y_Prueba))
#Ahora usando el algoritmo de regrecion lineal vemos que la precision es mas alta

#Eliminamos nuestra variable
del AlgoritmoRL

#Usamos regrecion de cresta (o curva)
#Importamos a una variable nuestroa algoritmo
AlgoritmoRidge = Ridge()

#Entrenamos a nuestro algoritmo
AlgoritmoRidge.fit(X_Entrenamiento, Y_Entrenamiento)

#Ahora veremos la precision de nuestro algoritmo
print(AlgoritmoRidge.score(X_Prueba, Y_Prueba))
#Vemos que los algoritmos de regresion lineal y de cresta tiene al mismo valor 