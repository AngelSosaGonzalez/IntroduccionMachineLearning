""" Arboles de decision (Regrecion): Este proyecto veremos sobre Arboles de decisiones, pero lo manejaremos en su forma de regresion
el proyecto de "IntroArbol.py" usa el algoritmo pero en la forma de clasificacion, cual es la diferencia entre una y otra, en pocas 
palabras regrecion te arroja un valor y clasificador caracteristicas de un objeto, por ejemplo regrecion arrojaria el precio de una casa
y el clasificador las caracteristicas de la casa, ahora antes de comenzar quiero aclarar que este proyecto se basa (o copia mas bien) del curso de 
Machine Learning del canal de: AprendeIA con Ligdi Gonzalez, fuente: https://www.youtube.com/watch?v=zvB0cshd0TM&list=PLJjOveEiVE4Dk48EI7I-67PEleEC5nxc3&index=24 """

#Importamos los modulos necesarios para el proyecto de arbol
#Importamos el DataSet (Como en los demas proyectos usaremos load_boston)
from sklearn import datasets

#Importamos Numpy para los Array
import numpy

#Importamos matplotlib para realizar graficas
import matplotlib.pyplot as plt

#Importamos el modulo que nos ayuda a separar los datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split

#Importamos nuestro algortimo de arbol
from sklearn.tree import DecisionTreeRegressor
#NOTA: los arboles se importan del modulo .tree de Sklearn

#Ahora importamos los datos del DataSet a una variable 
DatosBoston = datasets.load_boston()

#Ahora vamos a seleccionar los datos que requerimos del nuestro DataSet
X_Arbol = DatosBoston.data[:, numpy.newaxis, 5]

#Definimos los datos con sus etiquetas
Y_Arbol = DatosBoston.target

#Ya teniendo estos datos vamos a visualizarlos en una grafica de dispercion
plt.scatter(X_Arbol, Y_Arbol) #Recuerda scatter es para graficas de dispercion
plt.show()

#Clasificamos nuestros datos en entrenamiento y prueba
X_Entre, X_Prueba, Y_Entre, Y_Prueba = train_test_split(X_Arbol, Y_Arbol, test_size = 0.2)
""" NOTA: Cuando usamos un rango en especifico de datos utilizamos esos datos (por ejemplo en este
caso que usamos los datos y etiquetas de la columna 6), incertamos en los parametros de los datos 
en este caso lo llamamos X_Arbol y Y_Arbol, si queremos todo el DataSet pues solo llamamos a la 
variable que tiene el DataSet junto con '.data' para los datos y '.target' para las etiquetas de los datos """

#Definimos el algoritmo
ArbolRegre = DecisionTreeRegressor(max_depth=5) #Nota: max_depth es el numero de ramas que tendra nuestro arbol (concejo modificando este se puede evitar el sobreajuste)

#Entrenamos nuestro algoritmo
ArbolRegre.fit(X_Entre, Y_Entre)

#Calculamos el Score del algoritmo
print(ArbolRegre.score(X_Prueba, Y_Prueba)) #Datos de prueba (El que nos interesa)
print(ArbolRegre.score(X_Entre, Y_Entre)) #Datos de entrenamiendo (Para verificar si existe un sobre ajuste)
#Viendo los resultados, estos son algo optimos para el algoritmo aunque no son los mas adecuados


#Ahora vamos a graficar 
""" Con .arange lo que haremos es mostrar los valores consecutivos de un maximo y un minimo con un salto 
ejemplo: el maximo es 4 y el minimo es 1 con saltos de 1, nos dara un arreglo con los siguientes valores 
[4, 3, 2, 1] """
X_Comprimido = numpy.arange(min(X_Prueba), max(X_Prueba), 0.1)

""" Con la funcion reshape los acomodamos en una arreglo de X y Y dimenciones como por ejemplo
tenemos un arreglo de 3x1 seria asi [1, 2, 3], ahora con esta funcion acomodamos el arreglo a nuestro gusto
por ejemplo 1x3 que seria el mismo arreglo de datos pero ahora de forma vertical """
X_Comprimido = X_Comprimido.reshape((len(X_Comprimido), 1))

#Ahora ya teniend los datos organizados (o entendibles) los vamos a graficar
plt.scatter(X_Prueba, Y_Prueba)
plt.plot(X_Comprimido, ArbolRegre.predict(X_Comprimido), color = 'red', linewidth = 2)
plt.show()