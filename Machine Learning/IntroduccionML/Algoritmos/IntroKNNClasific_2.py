""" KNN Clasificacion: El algoritmo de vecinos más cercanos (KNN) es un tipo de algoritmos de aprendizaje automático supervisado. 
KNN es extremadamente fácil de implementar en su forma más básica, y sin embargo realiza tareas de clasificación bastante complejas. 
Si quieres saber mas aqui la fuente del proyecto: https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
antes de comenzar quiero aclarar que este proyecto se basa (o copia mas bien) del curso de 
Machine Learning del canal de: AprendeIA con Ligdi Gonzalez, fuente: https://www.youtube.com/watch?v=mPScafZY8co&list=PLJjOveEiVE4Dk48EI7I-67PEleEC5nxc3&index=42 """

#Comezamos a importar los modulos necesarios para este proyecto
#Importamos el DataSet, usaremos al data de Cancer
from sklearn.datasets import load_breast_cancer

#Importamos el modulo para separar los datos de entrenamiento y prueba
from sklearn.model_selection import train_test_split

#Importamos nuestro algoritmo KNN
from sklearn.neighbors import KNeighborsClassifier

""" Importamos metricas de rendimiento: 
- Matriz de confusión o error
- Precisión
- Recall o sensibilidad o TPR (Tasa positiva real)
- Exactitud
- Especificidad o TNR (Tasa negativa real)
- F1-Score """
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

#Iniciamos con meter nuestro DataSet a una variable
DatosCancer = load_breast_cancer()

#Despues separamos los datos de entrenamiento y prueba
X_Entrena, X_Prueba, Y_Entrena, Y_Prueba = train_test_split(DatosCancer.data, DatosCancer.target, test_size = 0.2)

""" Seleccionamos el algoritmo KNN
con respecto a este algortimo veremos que la funcion tiene 3 atributos:
- n_neighbors: Numero de vecinos a seleccionar
- metric: La métrica de distancia que se utilizará para el árbol
- p: Parámetro de potencia para la métrica Minkowski
Para mas informacion la documentacion del algoritmo: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html """
AlgoKNN = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)

#Entrenamos nuestro modelo
AlgoKNN.fit(X_Entrena, Y_Entrena)

#Realizamos una prediccion
Prediccion = AlgoKNN.predict(X_Prueba)

#Calculamos la score
print(AlgoKNN.score(X_Prueba, Y_Prueba))

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
