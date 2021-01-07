""" Naive Bayes: En un sentido amplio, los modelos de Naive Bayes son una clase especial de algoritmos de clasificación de Aprendizaje 
Automatico, o Machine Learning, tal y como nos referiremos de ahora en adelante. Se basan en una técnica de clasificación estadística llamada 
“teorema de Bayes”. Ahora ahora antes de comenzar quiero aclarar que este proyecto se basa (o copia mas bien) del curso de 
Machine Learning del canal de: AprendeIA con Ligdi Gonzalez, fuente: https://www.youtube.com/watch?v=P930ev-eyVk&list=PLJjOveEiVE4Dk48EI7I-67PEleEC5nxc3&index=49 """

#Vamos a comenzar con importar los modulos necesarios
#Primero comenzamos con el DataSet (El DataSet a utlizar ser relacionado con el cancer)
from sklearn.datasets import load_breast_cancer

#Importamos el modulo del separador de datos en datos de entrenamiento y prueba
from sklearn.model_selection import train_test_split

#Importamos el modulo de nuestro algortimo naive bayes
from sklearn.naive_bayes import GaussianNB

""" Importamos metricas de rendimiento: 
- Matriz de confusión o error
- Precisión
- Recall o sensibilidad o TPR (Tasa positiva real)
- Exactitud
- Especificidad o TNR (Tasa negativa real)
- F1-Score """
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score


#Comenzamos con el proyecto metiendo los datos del DataSet en una variable
DatosCancer = load_breast_cancer()

#Separamos los datos en datos de entrenamiento y prueba
X_Entrena, X_Prueba, Y_Entrena, Y_Prueba = train_test_split(DatosCancer.data, DatosCancer.target, test_size = 0.2) #Nota: data_size es el tamaño de la muestra respecto a data
#Para saber porque se usan data y targer te recomiento ver las caracteristicas del DataSet, se realiza con print(*Nombre de tu DataSet*.keys()) te arrojara las caracteriticas

#Vamos a definir nuestro algortimo
AlgoNaiveBayes = GaussianNB()

#Ya definido, realizamos el entrenamiento de nuestro algortimo 
AlgoNaiveBayes.fit(X_Entrena, Y_Entrena)

#Entrenado el algoritmo vamos a realizar una prediccion
Prediccion = AlgoNaiveBayes.predict(X_Prueba)

#Vamos a calcular el score de nuestro algoritmo
print(AlgoNaiveBayes.score(X_Prueba, Y_Prueba))
#Viendo el resultado del puntaje damos a la conclucion de que es un buen algoritmo ya que el score y la metricas son muy positivas

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