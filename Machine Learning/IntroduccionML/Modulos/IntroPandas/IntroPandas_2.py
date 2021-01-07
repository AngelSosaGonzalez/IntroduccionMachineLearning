""" Introduccion a Pandas: este proyecto veremos sobre el modulo de Pandas, veremos sobre el manejo de datos 
antes de comenzar veremos el concpeto de Pandas: 
Pandas es una herramienta de manipulación de datos de alto nivel desarrollada por Wes McKinney. 
Es construido con el paquete Numpy y su estructura de datos clave es llamada el DataFrame. 
El DataFrame te permite almacenar y manipular datos tabulados en filas de observaciones y columnas de variables. """

""" NOTA: En cursos que te encuentre sobre NumPy o Pandas al momento de importar los modulos te encontraras 
con la palabra reservada "as" esto es un alias, asi podemos invocar las funciones del modulo sin escribir todo el 
nombre por el momento para entender mejor el modulo no le pondremos alias """

#Importaremos la libreria de Pandas y Numpy
import pandas
import numpy

#Crear una DataFrame basica
#Crearemos el arreglo de datos con NumPy (Si no te acuerdas como checa el archivo "IntroNumPy")
DataFrame = numpy.array([[11, 12],[13, 14]])
#En forma de arreglo guardaremos en una variable el nombre de las columnas
Columnas = ['Colum No.1', 'Colum No.2']
#De igual forma que las columnas pero ahora con las filas
Filas = ['Fila No.1', 'Fila No.2']

""" Vamos a imprimir en consola el DataFrame creado, aunque igual lo podemos guardar en una variable y luego imprimirla.
Ahora para explicar la funcion de DataFrame de Pandas describire los atributos que nesesitamos para crear un DataFrame:
- data = Los datos que estaran dentro de nuestra DataFrame
- index = Nombre de las filas
- columns = Nombre de las columnas """
#Ejemplo de una DataFrame  
print(pandas.DataFrame(data = DataFrame, index = Filas, columns = Columnas )) 

#Otra forma de crear una DataFrame
""" Esta forma creamos el arreglo de datos NumPy dentro de la funcion DataFrame de Pandas 
a diferencia de la anterior las columnas y filas seran nombrasdas por numeros"""
DataFrame = pandas.DataFrame(numpy.array([[11, 12],[13, 14]]))
print(DataFrame)

#Series
#Crearemos una serie basica, la estructura de una serie en pandas es la misma a la de un JSON
SeriesBasica = pandas.Series({'Nombre':'Angel Sosa','Edad':'21','Ciudad':'Ecatepec'})
print(SeriesBasica)

#Funciones de Pandas
#Forma de nuestro DataFrame (el tamaño pues...)
#Creamos nuestra DataFrame, esto lo puedes verlo arriba de como se púeden crear
DataFrame = pandas.DataFrame(numpy.array([[11, 12],[13, 14]]))
#Con la funcion "shape" vamos a saber el tamaño de nuestro DataFrame
print('El tamaño de la DataFrame es: {}'.format(DataFrame.shape))
#Altura del DataFrame para esto usamos "len" funcio basica de Python y "Index" funcion de Pandas
print('La altura de nuestra DataFrame es: {}'.format(len(DataFrame.index)))

#Estadisticas de Pandas
#Descripcion
#Creamos nuestra DataFrame, esto lo puedes verlo arriba de como se púeden crear
DataFrame = pandas.DataFrame(numpy.array([[11, 12],[13, 14]]))
#Vamos a utilizar la funcion "describe" para crear estadisticas descriptivas
print('Estadisticas descriptiva')
print(DataFrame.describe())

#Descubrir la media del DataFrame usando la funcion "mean"
print('Media')
print(DataFrame.mean())

#Descubrir la correlacion del DataFrame
print('Correlacion')
print(DataFrame.corr())

#Contar los datos de nuestro DataFrame (conteo no suma)
print('Conteo de datos')
print(DataFrame.count())

#Valor mas alto de las columnas
print('El valor mas alto')
print(DataFrame.max())

#Valor mas bajo de las columnas
print('El valor mas bajo')
print(DataFrame.min())

#Mediana de nuestra DataFrame
print('Mediana de DataFrame')
print(DataFrame.median())

#Desviacion estandar
print('Desviacion estandar')
print(DataFrame.std())

#Seleccion
#Creamos nuestra DataFrame, esto lo puedes verlo arriba de como se púeden crear
DataFrame = pandas.DataFrame(numpy.array([[11, 12, 13],[13, 14, 15]]))
#Buscar las columnas de una DataFrame (para esto solo buscaremos por indice de la columna)
print('Los valores de la columna 0 es: ')
print(DataFrame[0])

#Buscaremos mas de una columna de nuestra DataFrame (Dato curioso: Podemos crear DataFrames de una DataFrame)
print('Los valores de la columna 0 y 2 es: ')
print(DataFrame[[0, 2]])

""" Buscar un valor con "iloc" tomado como referencia su posicion en fila y columna 
el primer corchete pertenece a las filas y el segundo a las columnas"""
print('El valor de la fila 0 y columna 1 es: {}'.format(DataFrame.iloc[0][1]))

#Buscar los valores de una fila
print('El valor de la fila 0 es: ')
print(DataFrame.loc[0])

#Otra forma de realizar la busqueda de filas
print('El valor de la fila 0 es: ')
print(DataFrame.iloc[0, :])

#Importacion y exportacion de datos 
#Abrir datos de un documento descargado (IMPORTANTE DEBE SER UN .csv)
DatosExport = pandas.read_csv('Datos\car.csv')

#Limpieza de datos
#Creamos nuestra DataFrame, esto lo puedes verlo arriba de como se púeden crear
DataFrame = pandas.DataFrame(numpy.array([[numpy.nan, 12, 13],[numpy.nan, 14, 15]]))
#Vamos a limpiar nuestro DataFrame con datos nulos 
print(DataFrame.isnull())

#Suma de valores nulos en nuestro DataFrame usando la funcion "sum"
print('Suma de valores nulos')
print(DataFrame.isnull().sum())

#Eliminar las filas nulas o perdidas
print(DataFrame.dropna())

#Eliminar las columnas nulas o con datos perdidos
print(DataFrame.dropna(axis= 1))

#Remplazar los valores nulos o perdidos, dentro de la funcion introduciremos porque lo vamos a remplazar
#En este ejemplo fue por una x
print(DataFrame.fillna('x'))