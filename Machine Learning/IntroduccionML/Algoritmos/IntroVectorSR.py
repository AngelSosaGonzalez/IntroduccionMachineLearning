""" Vectore de soporte de regresion: En este proyecto veremos en la marcha el concepto de como se puede aplicar este algoritmo de ML, 
en este proyecto aplicaremos los conociemiento basicos aprendidos en ML, de los antiguos proyecto que estan en este recopilatorio, 
te recomiendo revisar los proyecto que hablan sobre regrecion (Lineal, Multiple, Polinomial), antes de comenzar quiero aclarar que este proyecto se basa (o copia mas bien) del curso de 
Machine Learning del canal de: AprendeIA con Ligdi Gonzalez, fuente: https://www.youtube.com/watch?v=zvB0cshd0TM&list=PLJjOveEiVE4Dk48EI7I-67PEleEC5nxc3&index=24 """

#Importaremos los modulos necesarios para el proyecto
#Importamos Numpy para los arreglos
import numpy

#Importamos el modulo de matplotlib para graficar
import matplotlib.pyplot as plt

#Ya para el DataSet (casas de boston) importaremos el modulo correspondiente
from sklearn import datasets

#Importamos el modulo que nos ayuda a separar los datos de prueba a los de entrenamiento
from sklearn.model_selection import train_test_split

#Importamos el algoritmo a seleccionar (en este caso es VSR o SVR)
from sklearn.svm import SVR

#Importamos nuestro DataSet en una variable (Esto para poder manipular los datos, ya lo hemos hecho en antiguos proyectos)
BostonDatos = datasets.load_boston()
#NOTA: Puede imprimir el DataSet para verificar si los datos son los correctos

#Al igual que el proyecto de "IntroRegresionPoli.py" obtendremos los datos que queremos
#Ahora vamos a seleccionar los datos necesarios para esto usaremos la cantidad de habitaciones
X_VR = BostonDatos.data[:, numpy.newaxis, 5]

#Obtendremos las etiquetas de los datos
Y_VR = BostonDatos.target

#Graficamos los datos que obtuvimos de data y target, para esto usamos matplotlib
plt.scatter(X_VR, Y_VR) #Recuerda que scatter son para graficas de dispercion 
plt.show()

#Separamos los datos en entrenamiento y prueba
X_Entrena, X_Prueba, Y_Entrena, Y_Prueba = train_test_split(X_VR, Y_VR, test_size = 0.2) #Recuerda que test_size, es el tamaño de la muestra que obtendremos del DataSet 

""" Invocamos a nuestro algoritmo 
Atributos de la funcion de nuestro algoritmo:
- Kernel: Especificamos el tipo de datos a utilizar en nuestro algoritmo, como vimos en la grafica de dispercion nuestros datos 
son de tipo lineal, por lo que tenemos que especificarle a nuestro algoritmo que tipo de datos usamos
- C: Parámetro de regularización. La fuerza de la regularización es inversamente proporcional a C. 
Debe ser estrictamente positiva. La penalización es una penalización l2 al cuadrado.
- epsilon: Epsilon en el modelo epsilon-SVR. Especifica el tubo de épsilon dentro del cual no se 
asocia ninguna penalización en la función de pérdida de entrenamiento con puntos predichos dentro de una distancia epsilon desde el valor real.
Todo esto lo puedes leer en la documentacion de la funcion: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html """
AlgoSVR = SVR(kernel='linear', C=1.0, epsilon=0.2)

#Ya teniendo el algoritmo ya creado con sus atributos vamos a entrenarlo
AlgoSVR.fit(X_Entrena, Y_Entrena)

#Vemos el score que nos arroja nuestro algoritmo
print(AlgoSVR.score(X_Entrena, Y_Entrena)) #Usaremos lo datos de entrenamiento
print(AlgoSVR.score(X_Prueba, Y_Prueba)) #Ahora con los datos de prueba
#Vemos nos dio un resultamo muy bajo vamos a graficar para ver que cantidad de datos toma de muestra nuestro algoritmo

#Primero realizamos una prediccion
Y_Prediccion = AlgoSVR.predict(X_Prueba)

#Vamos a graficar, para este caso graficaremos igual que el algoritmo de regrecion polinomial
plt.scatter(X_Prueba, Y_Prueba)
plt.plot(X_Prueba, Y_Prediccion, color = 'red',  linewidth = 2) #Recuerda que plot sirve para la graficas de linea (Para este caso, nos mostrara los datos que recolecta nuestro algoritmo)
plt.show()
""" Ahora gracias a la grafica veremos que solamente se dibuja una linea esto porque en la creacion del algoritmo en el atributo del Kernel
seleccionamos lineal, por lo tanto solo se dibujara una linea y los datos que agarra la linea son lo que usaremos para la prediccion """ 

#Perooo... te preguntaras como podemos mejorar el algoritmo, cambiando parametros es la mas acertada, pero vamos a experimentar solamente usando el algoritmo sin modificar parametros
#Primero eliminamos la variable de nuestro algortimo (Esto para no cargar mucho el sistema)
del AlgoSVR

#Ahora lo volvemos a crear (Invocamos el algoritmo)
AlgoSVR = SVR()

#Entrenamos nuestro algortimo
AlgoSVR.fit(X_Entrena, Y_Entrena)

#Vemos el score que nos arroja nuestro algoritmo
print(AlgoSVR.score(X_Entrena, Y_Entrena)) #Usaremos lo datos de entrenamiento
print(AlgoSVR.score(X_Prueba, Y_Prueba)) #Ahora con los datos de prueba

#Realzamos una prediccion
Y_Prediccion = AlgoSVR.predict(X_Prueba)

#Graficamos para ver como esta nuestro algoritmo
plt.scatter(X_Prueba, Y_Prueba)
plt.plot(X_Prueba, Y_Prediccion, color = 'red', linewidth = 2)
plt.show()

""" Comparando los resultados veremos que el algorito aumento el score, no mucho, como digo podemos mejorarlo cambiando parametros, 
pero como usamos un kernel lineal una linea no agarra todos los datos necesarios para tener una buena prediccion """