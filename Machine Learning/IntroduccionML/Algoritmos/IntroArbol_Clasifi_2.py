""" Arbol de desiciones: Este programa veremos sobre arboles de desciones, pero ¿Que son?: Los árboles de decisión 
son una técnica de aprendizaje automático supervisado muy utilizada en muchos negocios. Como su nombre indica, 
esta técnica de machine learning toma una serie de decisiones en forma de árbol. Los nodos intermedios (las ramas) 
representan soluciones. Ahora ya entendido esto vamos con el codigo, antes de comenzar vamos a dar los creditos.
antes de comenzar quiero aclarar que este proyecto se basa (o copia mas bien) del curso de 
Machine Learning del canal de: AprendeIA con Ligdi Gonzalez, fuente: https://www.youtube.com/watch?v=_NU9jD303tM&list=PLJjOveEiVE4Dk48EI7I-67PEleEC5nxc3&index=52 """

#Antes de iniciar vamos a importar las librerias necesarias para el proyecto
#Importamos el DataSet
from sklearn.datasets import load_breast_cancer

#Importamos el separador de datos en datos de entrenamiento y prueba
from sklearn.model_selection import train_test_split

#Importamos el algoritmo deseado (Arbol de decision)
from sklearn.tree import DecisionTreeClassifier

""" Importamos metricas de rendimiento: 
- Matriz de confusión o error
- Precisión
- Recall o sensibilidad o TPR (Tasa positiva real)
- Exactitud
- Especificidad o TNR (Tasa negativa real)
- F1-Score """
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score


#Iniciamos con introducir los datos en una varible
DatosCancer = load_breast_cancer()

#Separamos los datos en datos de entrenamiento y de prueba
X_Entrena, X_Prueba, Y_Entrena, Y_Prueba = train_test_split(DatosCancer.data, DatosCancer.target, test_size=0.2)

#Ya separados los datos vamos a definir nuestro algoritmo
AlgoArbol = DecisionTreeClassifier(criterion='entropy') 
"""Nota: Función para medir la calidad de una división. Los criterios admitidos son "gini" para la impureza de Gini y la "entropy" para la ganancia de información.
Fuente: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier """

#Ya definido vamos a entrenarlo
AlgoArbol.fit(X_Entrena, Y_Entrena)

#Realizaremos una prediccion
Prediccion = AlgoArbol.predict(X_Prueba)

#Vamos a calcular el score
print(AlgoArbol.score(X_Prueba, Y_Prueba))

#**USANDO METRICAS**
""" Nota: Para usar las metricas dentro de la funcion de estas los dos argumento a seleccionar son:
- Prediccion
- Prueba (De los datos Y)
Ejemplo:
#Matrix de Exactitud
ExactMetric = accuracy_score(Y_Prueba, Prediccion)
print(ExactMetric)

Siempre debe ir primero el datos de prueba y despues la de prediccion

Si quieres sabes mas de las matrices de rendimiento 
info: https://www.youtube.com/watch?v=K5PNrX694HQ&list=PLJjOveEiVE4Dk48EI7I-67PEleEC5nxc3&index=36 """

#Matrix de Exactitud
ExactMetric = accuracy_score(Y_Prueba, Prediccion)
print(ExactMetric)

#Metrica de sensibilidad
SensiMetric = recall_score(Y_Prueba, Prediccion)
print(SensiMetric)

#Metrica de precision
PreciMetric = precision_score(Y_Prueba, Prediccion)
print(PreciMetric)

#Realizamos una metrica para saber el rendimiento del algoritmo
MatrixMetric = confusion_matrix(Y_Prueba, Prediccion)
print(MatrixMetric)
