""" Arbol de desiciones: Este programa veremos sobre arboles de desciones, pero ¿Que son?: Los árboles de decisión 
son una técnica de aprendizaje automático supervisado muy utilizada en muchos negocios. Como su nombre indica, 
esta técnica de machine learning toma una serie de decisiones en forma de árbol. Los nodos intermedios (las ramas) 
representan soluciones. Ahora ya entendido esto vamos con el codigo, antes de comenzar vamos a dar los creditos.
Para este proyecto nos basaremos en un curso dado por: AMP Tech, 
link: https://www.youtube.com/watch?v=269QJ5joMCc&list=PLA050nq-BHwMr0uk7pPJUqRgKRRGhdvKb&index=4 , tambien tiene 
una lista de reproduccion de solamente ML """

#Vamos a importar los modulos necesarios para nuestro proyecto
#Importamos el modulo para nuestro algoritmo (clasificacion)
from sklearn.tree import DecisionTreeClassifier

#Importaremos nuestros sets de datos, para el proyecto usaremos el set de dato de Iris y Cancer
from sklearn.datasets import load_breast_cancer, load_iris

#Importaremos nuestro separador de datos, este nos ayuda a separar datos de entrenamiento y prueba
from sklearn.model_selection import train_test_split

#Dentro del modulo de Sklearn importaremos un modulo especial para GraphViz (para graficar nuestro arbol)
from sklearn.tree import export_graphviz

#Importaremos nuestro modulo de GraphViz
import graphviz

#Importaremos le modulo de matplotlib esto para mostrar los graficos
import matplotlib.pyplot as plt

#NumPy para los arrays (nos servira para graficar)
import numpy


#Metemos los datos de nuestra dataset (primero usaremos Iris) en una variable
DatosIris = load_iris()

#Realizaremos la separacion de los datos entre Entrenamiento y Prueba usando la funcion de train_test_split(*data*, *target*)
X_Entrena, X_Prueba, Y_Entrena, Y_Prueba = train_test_split(DatosIris.data, DatosIris.target)

#Invocaremos nuestro algoritmo de arbol (con la funcion DecisionTreeClassifier())
AlgoArbol = DecisionTreeClassifier()

#Entrenamos nuestro algoritmo con nuestros datos de entrenamiento (X_Entrena y Y_Entrena)
AlgoArbol.fit(X_Entrena, Y_Entrena)

#Vamos a calcular el score (Con los datos de prueba y entrenamiento)
print(AlgoArbol.score(X_Prueba, Y_Prueba))
print(AlgoArbol.score(X_Entrena, Y_Entrena))

#Graficamos nuestro arbol en GraphViz (Creamos un archivo .dot)
export_graphviz(AlgoArbol, out_file='Datos\IntroArbol.dot', class_names=DatosIris.target_names, 
feature_names=DatosIris.feature_names, impurity=False, filled=True)

#Vamos a graficar nuestro arbol en GraphViz
with open('Datos\IntroArbol.dot') as AG:
    dot_graph = AG.read()
graphviz.Source(dot_graph)

#Ahora con matplotlib crearemos una grafica de barras para mostrar la importancia de las caracteristicas
CaracIris = DatosIris.data.shape[1]

""" Crearemos una grafica de barras lateral, feature_importances_ es una caracteristica del algoritmo de arbol para 
mostrar las caracteristicas importantes """
plt.barh(range(CaracIris), AlgoArbol.feature_importances_)

#Por ultimo caracteristicas de matplotlib para graficar
""" Datos o cracteristicas a mostrar (muestra una leyenda de esta)
arange(), muesta el intervalo de numero en un arreglo, en este caso el valor es 4
mostrara en un arreglo [0, 1, 2, 3], en base a este arreglo mostrara las leyendas
esto usando DatosIris.feature_names """
plt.yticks(numpy.arange(CaracIris), DatosIris.feature_names)

#Leyendas en los angulos X y Y
plt.xlabel('Importancia')
plt.ylabel('Caracteristicas')

#Mostramos la grafica
plt.show()

""" Arreglaremos nuestro algortimo ya que tiene porcentajes muy elevados en puntacion 
esto aunque suene bueno se tiene el termino de "sobre-ajuste", Se denomina sobreajuste 
al hecho de hacer un modelo tan ajustado a los datos de entrenamiento que haga que no 
generalice bien a los datos de test. """

#Primero cambiamos la profundidad de nuestro arbol (los niveles que tendra), en el curso usan 3 asi que lo haremos igual
AlgoArbol = DecisionTreeClassifier(max_depth=3)

#Entrenamos
AlgoArbol.fit(X_Entrena, Y_Entrena)

#Por ultimo calculamos los score para ver si se resolvio el problema
print(AlgoArbol.score(X_Prueba, Y_Prueba))
print(AlgoArbol.score(X_Entrena, Y_Entrena))
#Vemos que los resultados son los deseados (menos de 1, pero arriba del umbral del 8 o 9)