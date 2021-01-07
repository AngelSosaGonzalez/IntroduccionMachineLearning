""" Regresion lineal multiple: Este proyecto aplicaremos el algoritmo de RLM (regresion lineal multiple), para esto aplicaremos
el mismo caso que el anterior algoritmo (IntroRegresionLineal.py, esta en la misma carpeta), que son los datos en la casas de boston
usaremos el anterior Data de Boston, antes de comenzar quiero aclarar que este proyecto se basa (o copia mas bien) del curso de 
Machine Learning del canal de: AprendeIA con Ligdi Gonzalez, fuente: https://www.youtube.com/watch?v=SZyH6YkQqIk&list=PLJjOveEiVE4Dk48EI7I-67PEleEC5nxc3&index=15 """

#Importaremos los modulos necesarios
#Importamos la libreria de SKlearn y los DataSet de SkLearn (Algoritmo de regresion y datos)
from sklearn import linear_model, datasets
#Para dividir nuestra Data en pruebas y entrenamiento
from sklearn.model_selection import train_test_split


#Importamos nuestros datos y los guardamos en una variable
DatosBoston = datasets.load_boston()

#Crearemos el predictor para nuestro algoritmo, osea del DataSet seleccionamos los datos que requerimos (POR LO QUE USAREMOS LAS COLUMNAS 5, 6, 7)
X_DatosHabitRLM =  DatosBoston.data[:, 5:8]

#Definimos las columnas pertenecientes a los datos que vamos a obtener
Y_DatosHabitRLM = DatosBoston.target

""" Vamos a ocupar la funcion para el entrenamiento y pruebas (train_test_split) 
para el uso de esta funcion lo dividiremos en 4 variables donde se guardara los 
valores de entrenamiento y purebas, para esto los dividiremos en X y Y (para pruebas y entrenamiento)n
nota: el test_size es el tamaño de la prueba que obtendremos """
X_Entrenamiento, X_Prueba, Y_Entrenamiento, Y_Prueba = train_test_split(X_DatosHabitRLM, Y_DatosHabitRLM, test_size = 0.2)

#Importaremos nuestro algortimo (Para RL y RLM se importa el mismo algoritmo)
RLM  = linear_model.LinearRegression()

#Entrenamos nuestro algortimo
RLM.fit(X_Entrenamiento, Y_Entrenamiento)

#Ahora calculamos nuestro score (usamos los datos de entrenamiento)
print(RLM.score(X_Entrenamiento, Y_Entrenamiento))
#Como vemos en nuestro Score el procentaje es de 50, es buen porcentaje pero no es el mas optimo (tendremos que cambiar el algoritmo)