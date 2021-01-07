""" Algoritmo de regrecion: En estadísticas, regresión lineal es una aproximación para modelar la relación entre una variable escalar dependiente “y” y una o mas variables explicativas nombradas con “X”.
En este proyecto veremos el uso de este algoritmo para Machine Learning, para este proyecto vamos a a basarnos en el curso de Machine Learning del canal
de: AprendeIA con Ligdi Gonzalez, fuente: https://www.youtube.com/watch?v=SZyH6YkQqIk&list=PLJjOveEiVE4Dk48EI7I-67PEleEC5nxc3&index=15 """

#Importaremos NumPy para usar arreglos
import numpy as np
#Importamos la libreria de SKlearn y los DataSet de SkLearn (Algoritmo de regresion y datos)
from sklearn import linear_model, datasets
#Importamos matplotlib para graficar
import matplotlib.pyplot as plt
#Para dividir nuestra Data en pruebas y entrenamiento
from sklearn.model_selection import train_test_split

#Importamos nuestros datos y los guardamos en una variable
DatosBoston = datasets.load_boston()

#Seleccionamos la columna 5 de nuestro DataSet (El x en nuestra formula) 
X = DatosBoston.data[:, np.newaxis, 5]

#Seleccionamos los datos correspondiente a las etiquetas
Y = DatosBoston.target

#Ahora vamos a graficar los datos obtenidos
plt.scatter(X, Y)
plt.xlabel('Num. Habitaciones')
plt.ylabel('Media de habitaciones')
plt.show()

""" Vamos a ocupar la funcion para el entrenamiento y pruebas (train_test_split) 
para el uso de esta funcion lo dividiremos en 4 variables donde se guardara los 
valores de entrenamiento y purebas, para esto los dividiremos en X y Y (para pruebas y entrenamiento) """
X_Entrenamiento, X_Prueba, Y_Entrenamiento, Y_Prueba = train_test_split(X, Y, test_size = 0.2)

#Importaremos el modelo de regrecion lineal
AlgRegrecionLineal = linear_model.LinearRegression()

#Entrenamos a nuestro algoritmo con nuestros datos
AlgRegrecionLineal.fit(X_Entrenamiento, Y_Entrenamiento)

#Hacemos una prediccion
Prediccion = AlgRegrecionLineal.predict(X_Prueba)

#Calculamos la presicion de nuestro algoritmo
print(AlgRegrecionLineal.score(X_Entrenamiento, Y_Entrenamiento))

#Graficamos los resultados
#Para realizar una objeto de dispercion
plt.scatter(X_Prueba, Y_Prueba)
#Para realzar un objeto lienal (Para saber que tantos tantos datos vamos a obtener)
plt.plot(X_Prueba, Prediccion, color = 'red', linewidth = 3)
plt.show()

""" Conclucion: En lo visto en el Score y en la grafica vemos que la presicion del algoritmo de regrecion lineal no es una algoritmo 
optimo para nuestros datos """