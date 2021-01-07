""" Estimadores de Incertidumbre: Este consiste en una funcion que esta incluida en algunos algoritmos del modulo,
lo que hace es darnos una probabilidad de que tan seguro esta nuestro algoritmo para realizar una prediccion,
Para este proyecto nos basaremos en un curso dado por: AMP Tech, 
link: https://www.youtube.com/watch?v=2A7Hz3RjhIY&list=PLA050nq-BHwMr0uk7pPJUqRgKRRGhdvKb&index=6, tambien tiene 
una lista de reproduccion de solamente ML """

#Comenzamos creando un proyecto usando SVM (Para utilizar SVC, otro algoritmo)
#Importamos el nuestro DataSet (Usaremos el de Iris)
from sklearn.datasets import load_iris

#Importamos el separador de datos en datos de entrenamiento y prueba
from sklearn.model_selection import train_test_split

#Importamos el algoritmo 
from sklearn import svm

#Importamos los datos del DataSet a una variable
DatosIris = load_iris()

#Ahora separamos los datos en datos de entrenamiento y prueba
X_Entrena, X_Prueba, Y_Entrena, Y_Prueba = train_test_split(DatosIris.data, DatosIris.target)

#Vamos a seleccionar nuestro algoritmo
AlgoSvm = svm.SVC(probability= True)

#Entrenamos nuestro algoritmo
AlgoSvm.fit(X_Entrena, Y_Entrena)

#Ahora utilizamos las funciones de incertidumbre
""" En primer lugar sabremos a que seccion pertenece cada muestra (en este caso en que tipo de iris pertenece), pero..., ¿Como 
lo sabremos?, es facil en la DataSet tenemos 3 secciones: 'setosa', 'versicolor', 'virginica', al momento de realizar la funcion
veremos un arreglo de 3 numeros por muestra, el numero mas alto es al grupo que pertence, para entenderlo mas facil, lo que haremos
es impirmir el nombre de las etiquetas como guias """

#Nombre de las etiquetas
print(DatosIris.target_names)

""" Usamos la funcion de Estimadores, los argumento a usar para la funcion solamente es escribir los datos de prueba
y los datos en los corchetes es el numero de muestras a mostrar """
print(AlgoSvm.decision_function(X_Prueba)[:10])

#Ahora lo veremos en probabilidad
#Nombre de las etiquetas
print(DatosIris.target_names)

#Usaremos los mismos argumentos
print(AlgoSvm.predict_proba(X_Prueba)[:10])

""" Ahora lo veremos pero en el valor de las etiquetas, donde en vez de usar el nombre vamos a ver el grupo a que pertence 
donde: 
- 0: setosa
- 1: versicolor
- 2: virginica """
#Usaremos los mismos argumentos
print(AlgoSvm.predict(X_Prueba)[:10])