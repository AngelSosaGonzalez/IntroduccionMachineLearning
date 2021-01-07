""" En este proyecto veremos lo que es NumPy, comenzaremos con una pequeña introduccion de este, como 
los arreglos entre otras cosas """

""" NOTA: En cursos que te encuentre sobre NumPy o Pandas al momento de importar los modulos te encontraras 
con la palabra reservada "as" esto es un alias, asi podemos invocar las funciones del modulo sin escribir todo el 
nombre por el momento para entender mejor el modulo no le pondremos alias """

#Importamos el modulo 
import numpy

#Arreglos
#Crear un arreglo dimensional
ArregloDi = numpy.array(['a', 'b', 'c'])
print('Esto es un arreglo dimensional: {}'.format(ArregloDi))

#Ahora vamos con los bidimensional
ArregloBi = numpy.array([('a', 'b', 'c'), (1, 2, 3)])
print('Esto es un arreglo bidimensional: {}'.format(ArregloBi))

#Matrices
#Crear una matriz de 3*4 con solo 1
MatrizUnos = numpy.ones((3, 4))
print(MatrizUnos)

#Crear una matriz de 3*4 con solo 0
MatrizCero = numpy.zeros((3, 4))
print(MatrizCero) 

#Crear una matriz de 3*4 con valores aleatorios
MatrizAlea = numpy.random.random((3, 4))
print(MatrizAlea)

#Crear una matriz de 3*4 vacio
MatrizVacia = numpy.empty((3, 4))
print(MatrizVacia)

#Crear una matriz de 3*4 con un valor repetido
MatrizNum = numpy.full((3, 4), 5)
print(MatrizNum)

#Crear una matriz de 3*4 con valores incrementales ENTEROS
MatrizIncEnt = numpy.arange(0, 10, 1)
print(MatrizIncEnt)

#Crear una matriz de 3*4 con valores incrementales DECIMALES
MatrizIncDec = numpy.linspace(0, 3, 5)
print(MatrizIncDec)

#Crear una matriz de identidad de 4*4 (Esta forma das el tamaño de Filas * Columnas)
MatrizIdenti = numpy.eye(4, 4)
print(MatrizIdenti) 

#Crear una matriz de identidad (Esta forma das el tamaño de los "1" en la matriz)
MatrizIdenti = numpy.identity(3)
print(MatrizIdenti)

#Conocer la dimecion de una matriz dimensional o bidimencional
#Dimencional
MatrizDim = numpy.array([1, 2, 3])
print('Las dimesiones de esta matriz es: {}'.format(MatrizDim.ndim))

#Bidimencional
MatrizBidim = numpy.array([(1, 2, 3), (4, 5, 6)])
print('Las dimesiones de esta matriz es: {}'.format(MatrizBidim.ndim))

#Saber los tipos de datos
TipoDato = numpy.array([1, 2, 3,])
print(TipoDato.dtype)

#Concer el tamaño de la matriz
MartizTamaño = numpy.array([(1, 2, 3), (4, 5, 6)])
print(MartizTamaño.size)

#Conocer la forma de la matriz
MatrizForma = numpy.array([(1, 2, 3), (4, 5, 6)])
print(MatrizForma.shape)

#Cambio de forma en matrices
MatrizIncial = numpy.array([(1, 2, 3), (4, 5, 6)])
print('La matriz incial es: {}'.format(MatrizIncial))
#Vamos a cambiar de forma
MatrizCambio = MatrizIncial.reshape(3, 2)
print('La matriz reescalada es: {}'.format(MatrizCambio))

#Seleccionar un dato de nuestra matriz
Seleccion = numpy.array([(1, 2, 3), (4, 5, 6)])
print('El dato en la localizacion (0,1) es: {}'.format(Seleccion[0,1]))

#Seleccionar mas de un dato de nuestra matriz
Seleccion = numpy.array([(1, 2, 3), (4, 5, 6)])
print('Los datos de la fila 0 es: {}'.format(Seleccion[0:, 0]))
#NOTA: El valor de ",0" es el valor de la columna selecionada 

#Encontrar valor maximo de una matriz
Maximo = numpy.array([(1, 2, 3), (4, 5, 6)])
print('El valor maximo de la matriz es: {}'.format(Maximo.max()))

#Encontrar el valor minimo
Minimo = numpy.array([(1, 2, 3), (4, 5, 6)])
print('El valor minimo de la matriz es: {}'.format(Minimo.min()))

#Suma de una fila 
SumaFila = numpy.array([1, 2, 3])
print('La suma de la fila completa es: {}'.format(SumaFila.sum()))

#Raiz cuadrada de una matriz
MatrizRaiz = numpy.array([(1, 2, 3), (4, 5, 6)])
print('La raiz cuadrada de la matriz es: {}'.format(numpy.sqrt(MatrizRaiz)))

#Desviacion estandar 
MatrizDesvi = numpy.array([(1, 2, 3), (4, 5, 6)])
print('La desviacion estandar de la matriz es: {}'.format(numpy.std(MatrizDesvi)))

#Operaciones con matrices 
MatrizUno = numpy.array([(1, 2, 3), (4, 5, 6)])
MatrizDos = numpy.array([(1, 2, 3), (4, 5, 6)])

#Suma
print('La suma de matriz 1 y matriz 2 es: {}'.format(MatrizUno + MatrizDos))

#Resta
print('La resta de matriz 1 y matriz 2 es: {}'.format(MatrizUno - MatrizDos))

#Multiplicacion
print('La multiplicacion de matriz 1 y matriz 2 es: {}'.format(MatrizUno * MatrizDos))

#Division
print('La division de matriz 1 y matriz 2 es: {}'.format(MatrizUno / MatrizDos))