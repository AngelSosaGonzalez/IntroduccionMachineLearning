""" Bosque aleatorios (Clasificacion): En este pryecto veremos sobre el algoritmo de bosques aleatorios, pero antes que nada veremos que son los bosques
aleatorios: Random forest también conocidos en castellano como '"Bosques Aleatorios"' es una combinación de árboles predictores tal que cada árbol 
depende de los valores de un vector aleatorio probado independientemente y con la misma distribución para cada uno de estos. Ya teniendo en cuenta el 
concepto vamos a darle, antes de comenzar quiero aclarar que este proyecto se basa (o copia mas bien) del curso de 
Machine Learning del canal de: AprendeIA con Ligdi Gonzalez, fuente:https://www.youtube.com/watch?v=Me7DSNW0pI8&list=PLJjOveEiVE4Dk48EI7I-67PEleEC5nxc3&index=55 """

#Anter de iniciar vamos a importar los modulos necesarios
#Importamos nuestra DataSet (DataSet de Cancer)
from sklearn.datasets import load_breast_cancer

#Importamos el separador de datos en datos de entrenamiento y prueba
from sklearn.model_selection import train_test_split

#Importamos nuestro algortimo
from sklearn.ensemble import RandomForestClassifier

""" Importamos metricas de rendimiento: 
- Matriz de confusión o error
- Precisión
- Recall o sensibilidad o TPR (Tasa positiva real)
- Exactitud
- Especificidad o TNR (Tasa negativa real)
- F1-Score """
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

#Comenzamos con introducir  los datos en una variable
DatosCancer = load_breast_cancer()

#Ahora separamos los datos en datos de entrenamiento y datos de prueba
X_Entre, X_Prueba, Y_Entre, Y_Prueba = train_test_split(DatosCancer.data, DatosCancer.target, test_size = 0.2)

#Ya teniendo los datos vamos a seleccionar nuestro algoritmo
BosqueAlgo = RandomForestClassifier(n_estimators=10, criterion='entropy') # n_estimator: Numero de arboles que se utilizara
"""Nota: Función para medir la calidad de una división. Los criterios admitidos son "gini" para la impureza de Gini y la "entropy" para la ganancia de información.
Fuente: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier """

#Entrenamos nuestro algortimo
BosqueAlgo.fit(X_Entre, Y_Entre)

#Entrenamos realizaemos una prediccion
Prediccion = BosqueAlgo.predict(X_Prueba)

#Veremos el score de nuetros algoritmo
print(BosqueAlgo.score(X_Prueba, Y_Prueba))

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