""" Bosque aleatorios (Regresion): En este pryecto veremos sobre el algoritmo de bosques aleatorios, pero antes que nada veremos que son los bosques
aleatorios: Random forest también conocidos en castellano como '"Bosques Aleatorios"' es una combinación de árboles predictores tal que cada árbol 
depende de los valores de un vector aleatorio probado independientemente y con la misma distribución para cada uno de estos. Ya teniendo en cuenta el 
concepto vamos a darle, antes de comenzar quiero aclarar que este proyecto se basa (o copia mas bien) del curso de 
Machine Learning del canal de: AprendeIA con Ligdi Gonzalez, fuente:https://www.youtube.com/watch?v=E2u-VxSXPXc&list=PLJjOveEiVE4Dk48EI7I-67PEleEC5nxc3&index=31 """

#Importaremos los modulos necesarios
#Comezamos con importar la DataSet
from sklearn.datasets import load_boston

#Importamos NumPy para los array
import numpy

#Importamos matplotlib para graficar
import matplotlib.pyplot as plt

#Importamos el separador de datos en datos de entrenamiento y prueba
from sklearn.model_selection import train_test_split

#Importamos el algoritmo (Bosque aleatorios)
from sklearn.ensemble import RandomForestRegressor

#Metemos la DataSet a una variable para poder utilizarla 
BostonDatos = load_boston()

#Obtrendremos de la DataSet la columna 6
X_Boston = BostonDatos.data[:, numpy.newaxis, 5] #NOTA: numpy.newaxis sirve para separar los datos de un arreglo en direferntes
""" Ejemplo de newaxis: 
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 
>>> a[:, np.newaxis]
array([[0],
       [1],
       [2],
       [3],
       [4],
       [5],
       [6],
       [7],
       [8],
       [9]]) """

#Ahora de los datos obtenidos sacaremos sus etiquetas
Y_Boston = BostonDatos.target

#Ahora veremos los datos en una grafica (de dispercion)
plt.scatter(X_Boston, Y_Boston) #Los argumentos de la funcion son los datos que separamos y metimos a las variables
plt.show()

#Ahora vamos a separar los datos en datos de entrenamiento y prueba 
X_Entrena, X_Prueba, Y_Entrena, Y_Prueba = train_test_split(X_Boston, Y_Boston, test_size = 0.2) #Nota: test_size es el tamaño de la muestra

""" Seleccionamos a nuestro algoritmo (Bosques aleatorios) donde los parametros a modificar de la funcion son:
- n_estimators: El numero de arboles a crear (Por defecto son 10)
- max_depth: Como en los arboles de desicion es el maximo de ramas o profundidad de nuestro arbol (o arboles)
para los demas argumento o informacion de este algoritmo en la documentacion: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html """
AlgoBosques = RandomForestRegressor(n_estimators=300, max_depth=8) #Nota: Entre mas arboles mejor sera nuestro modelo, pero mas lento al momento de utilizarlo

#Entrenamos nuestro algoritmo
AlgoBosques.fit(X_Entrena, Y_Entrena)

#Ahora calculamos el score de nuestro algoritmo
print(AlgoBosques.score(X_Prueba, Y_Prueba)) #Con datos de prueba
print(AlgoBosques.score(X_Entrena, Y_Entrena)) #Con datos de entrenamiento
#Viendo los resultados los porcentajes son aceptables, mas no los mas adecuados (Esto puede que lo podemos mejorar cambiando parametros o cambiando el algoritmo)

#Graficaremos el algoritmo
X_Comprimido = numpy.arange(min(X_Prueba), max(X_Prueba), 0.1)
X_Comprimido = X_Comprimido.reshape((len(X_Comprimido), 1))
#Nota: si no conoces las funciones de '.arange', '.reshape', te recomiendo ver el proyecto de 'IntroArbolRegre.py', ahi lo explico

#Ahora ya teniend los datos organizados (o entendibles) los vamos a graficar
plt.scatter(X_Prueba, Y_Prueba)
plt.plot(X_Comprimido, AlgoBosques.predict(X_Comprimido), color = 'red', linewidth = 2)
plt.show()