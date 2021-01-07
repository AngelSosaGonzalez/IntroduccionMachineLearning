""" Introduccion a la clasificacion: Este proyecto pondrmos ya a prueba lo aprendido en los otros proyectos de introduccion a los modulos 
encargados en Machine Learning, pero ahora no enfocaremos en la clasificacion pero antes de esto, ¿Que es clasificacion?: En pocas palabras un sistema de clasificación predice una categoría 
ya sabiendo esto vamos con el codigo """

""" Para este proyecto nos basaremos en un curso dado por: AMP Tech 
Fuente (o video): https://www.youtube.com/watch?v=hzOCDgfsSSQ&list=PLA050nq-BHwMr0uk7pPJUqRgKRRGhdvKb&index=2&pbjreload=101
digo esto porque usaremos una clasificacion de sklearn de iris """

#Importacion de los modulos requeridos
#NumPy (Arreglos)
import numpy as np
#Sklearn (Algoritmos de clasificacion)
import sklearn
#DataSet de las Iris (la flor)
from sklearn.datasets import load_iris
#Para dividir nuestra Data en pruebas y entrenamiento
from sklearn.model_selection import train_test_split
#Importaremos un algoritmo de Sklearn de KNN o vecinos cercanos
from sklearn.neighbors import KNeighborsClassifier

#Llamamos al set de datos (Iris)
FlorIris = load_iris()
#NOTA (y dato curioso que comparte el curso): El tipo de dato de nuestro DataSet es de tipo Bunch que se parece a los diccionarios

#Vamos a conocer las llaves (o las columnas de nuestro DataSet)
print(FlorIris.keys())

""" Para conocer el contenido de datos de cada llave solo basta con llamarlo la variable + el nombre de la llave como este ejemplo. """
#Esto arrojara una matriz de datos de cada flor iris obtenida
print(FlorIris['data'])

#Este arrojara los tipos de Flores que se obtuvieron (o las etiquetas)
print(FlorIris['target'])

#Este arrojara el nombre de las etiquetas 
print(FlorIris['target_names'])

#Este arroja en el nombre al que pertenece los datos
print(FlorIris['feature_names'])

""" Vamos a ocupar la funcion para el entrenamiento y pruebas (train_test_split) 
para el uso de esta funcion lo dividiremos en 4 variables donde se guardara los 
valores de entrenamiento y purebas, para esto los dividiremos en X y Y (para pruebas y entrenamiento) """
X_Entrenamiento, X_Prueba, Y_Entrenamiento, Y_Prueba = train_test_split(FlorIris['data'], FlorIris['target'])

""" Vamos a darle un valor a nuestra K, esto significa cuantos vecinos vamos a tomar como referencia
para saber al momento de ingresar un nuevo registro o clasificar un registro nuevo, este toma como referencia el 
numero de vecino introducidos (osea el valor de K) al momento de darle un valor este tomara el valor 
de los vecinos que el nuevo registro tomara de referencia para asi clasificarlo """
#Como en el curso toma 7 vecino igual tomaremos 7 vecinos
K = KNeighborsClassifier(n_neighbors=7)

#Ahora vamor a entrenar nuestro clasificador, para entrenarlo usaremos los valores de X_Entrenamiento y Y_Entrenamiento, usando la funcion "fit"
K.fit(X_Entrenamiento, Y_Entrenamiento)

#Calcularemos la presicion de nuestro clasificador
print(K.score(X_Prueba, Y_Prueba))

#Por ultimo probamos, para esto le mandamos un arreglo bidimencional para que prediga los datos, este arreglo lo llenaremos con los valores basados en la llave de 'data' y usando la funcion 'predict'
print(K.predict([[6.2, 3.1, 5.3, 2.4]]))
#Los datos que le arrojamos se acerca a los virginica o de etiqueta '2'

