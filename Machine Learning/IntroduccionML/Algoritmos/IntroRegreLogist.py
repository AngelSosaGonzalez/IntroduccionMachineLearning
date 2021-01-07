""" Regresion logistica: La Regresión Logística es un método estadístico para predecir clases binarias. El resultado o variable objetivo es de 
naturaleza dicotómica. Dicotómica significa que solo hay dos clases posibles. Por ejemplo, se puede utilizar para problemas de detección de 
cáncer o calcular la probabilidad de que ocurra un evento. Fuente: https://aprendeia.com/regresion-logistica-multiple-machine-learning-teoria/
Ahora conociendo esto coemzaremos a programar usando este algoritmo antes de comenzar quiero aclarar que este proyecto se basa (o copia mas bien) del curso de 
Machine Learning del canal de: AprendeIA con Ligdi Gonzalez, fuente: https://www.youtube.com/watch?v=rUHZb_TzWVs&list=PLJjOveEiVE4Dk48EI7I-67PEleEC5nxc3&index=39 """

#Comezaremos importando los modulos necesarios para el proyecto
#Importamos NumPy para los arreglos
import numpy

#Primero importamos nuestro DataSet
from sklearn.datasets import load_breast_cancer

#Importamos nuestro separador de datos en datos de entrenamiento y prueba
from sklearn.model_selection import train_test_split

#Importamos el modulo de escalamiento de datos
from sklearn.preprocessing import StandardScaler

#Importamos nuestro algortimo
from sklearn.linear_model import LogisticRegression

""" Importamos metricas de rendimiento: 
- Matriz de confusión o error
- Precisión
- Recall o sensibilidad o TPR (Tasa positiva real)
- Exactitud
- Especificidad o TNR (Tasa negativa real)
- F1-Score """
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

#Comenzamos con poner nuestro DataSet en una variable
CancerData = load_breast_cancer()

#Separamos nuestros datos en datos de entrenamiento y prueba
X_Entrena, X_Prueba, Y_Entrena, Y_Prueba = train_test_split(CancerData.data, CancerData.target, test_size = 0.2)

#Realizamos un escalado de datos
Escalado = StandardScaler()
#Escalamos nuestros datos X (Entrenamiento y prueba)
X_Entrena = Escalado.fit_transform(X_Entrena)
X_Prueba = Escalado.fit_transform(X_Prueba)

#Seleccionamos nuestro algoritmo
AlgoRegreL = LogisticRegression()

#Entrenamos nuestro algoritmo
AlgoRegreL.fit(X_Entrena, Y_Entrena)

#Realizamos una prediccion
Prediccion = AlgoRegreL.predict(X_Prueba)

#Calculamos el score
print(AlgoRegreL.score(X_Prueba, Y_Prueba))
#Vemos que en base al puntaje de nuestro Score vemos que el algoritmo es un buen algortimo

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


