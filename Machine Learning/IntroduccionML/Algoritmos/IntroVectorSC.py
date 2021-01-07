""" SVC: El SVM, por sus siglas en inglés, construye un hiperplano en un espacio multidimensional para separar 
las diferentes clases. El SVM genera un hiperplano óptimo de forma iterativa, que se utiliza para minimizar un error. 
La idea central de SVM es encontrar un hiperplano marginal máximo que mejor divida el conjunto de datos en clases. 
ahora antes de comenzar quiero aclarar que este proyecto se basa (o copia mas bien) del curso de 
Machine Learning del canal de: AprendeIA con Ligdi Gonzalez, fuente:https://www.youtube.com/watch?v=sJcYUmseGJQ&list=PLJjOveEiVE4Dk48EI7I-67PEleEC5nxc3&index=46 """

#Antes de comenzar vamos a importar los modulos necesarios
#Importaremos el DataSet (Usaremos el de Cancer)
from sklearn.datasets import load_breast_cancer

#Imprtamos el separador de datos de entrenamiento y prueba
from sklearn.model_selection import train_test_split

#Importamos el algoritmo 
from sklearn.svm import SVC

""" Importamos metricas de rendimiento: 
- Matriz de confusión o error
- Precisión
- Recall o sensibilidad o TPR (Tasa positiva real)
- Exactitud
- Especificidad o TNR (Tasa negativa real)
- F1-Score """
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

#Inicamos metiendo nuestro DataSet a una variable
DatosCancer = load_breast_cancer()

#Ahora separamos nuestros datos en entrenamiento y prueba
X_Entre, X_Prueba, Y_Entre, Y_Prueba = train_test_split(DatosCancer.data, DatosCancer.target, test_size = 0.2)

#Seleccionamos nuestro algortimo
""" Para mas informacion del kernel:
- https://www.youtube.com/watch?v=YO1yAEKDA64&list=PLJjOveEiVE4Dk48EI7I-67PEleEC5nxc3&index=44
- https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html """
AlgoSVC = SVC(kernel='linear')

#Vamos a entrenar nuestro algortimo 
AlgoSVC.fit(X_Entre, Y_Entre)

#Realizamo una predicicon
Prediccion = AlgoSVC.predict(X_Prueba)

#Calculalos el score
print(AlgoSVC.score(X_Prueba, Y_Prueba))

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