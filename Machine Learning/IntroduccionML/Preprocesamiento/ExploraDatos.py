""" Exploracion de los datos: En este proyecto veremos sobre la exploracion de los datos, con exploracion nos referimos pues como dice el verbo
explorar todo el contenido de los datos, para el proyecto usaremos el DataSet que guardamos en el proyecto de "ImportExportDatos.py", el 
DataSet con el nombre "DataFrameEdit.csv" 
Antes de comenzar quiero aclarar que este proyecto se basa (o copia mas bien) del curso de Machine Learning del canal de: AprendeIA con Ligdi Gonzalez, 
fuente:https://www.youtube.com/watch?v=nWdgf7-VOBE&list=PLJjOveEiVE4BK9Vnnl99H2IlYGhmokn1V&index=3 """

#Antes de comenzar con el proyecto importaremos los modulos necesarios
import pandas as pd #Rescuerda que "as" es para poner un alias al modulo

#Importamos nuestros CSV (Para este caso no usare una variable para la URL, lo declarare directamente)
DataFrameTitanic = pd.read_csv("IntroduccionML/Preprocesamiento/Data/DataFrameEdit.csv")

#Probamos que nuestra Data fue importado con exito viendo
print(DataFrameTitanic.head())

#Ya con el modulo importado ahora veremos que tipos de datos tiene nuesto DataSet (O DataFrame)
print(DataFrameTitanic.dtypes)
""" Explicacion: En nuestra Data vemos 3 tipos de datos:
- int64 (Es lo mismo que int): Se refiere a numero enteros
- float: Numero con coma flotante (con decimales pues...)
- object: Cadena de caracteres
Pero tambien existen los datetime que consta de datos de tipo fecha, para usar nuestra Data para los algritmo de ML necesitamos datos del tipo 
numerico (int, float), pero en esta practica no lo veremos """

#Ahora veremos la informacion de nuestra Data de la manera mas estadistica
print(DataFrameTitanic.describe())
""" Donde:
- count: Es el conteo de datos en la columna 
- mean: El el valor de la media de la columna
- std: Desviacion estandar
- min: Valor minimo de la columna
- 25%: |
- 50%: |-> Los limites de la columna
- 75%: |
- max: Valos maximo de la columna
En la consola de Python veremos que no estan todas las columnas, esto se debe a que solamente describe los valores numerico y tomo las columnas con
valores numericos """

#Veremos la misma informacion estadistica pero ahora con todas la columnas
print(DataFrameTitanic.describe(include = 'all'))
""" Donde veremos que se agregaron mas columnas:
- unique: Numero de objetos distintos en la columna
- top: El datos que mas aparece
- freq: Numero de veces que aparece un objeto 'top'
Nota: El termino Nan significa 'No es un numero' """

#Al igual que la funcion '.head()' y '.tail()' esta nos muestra los primeros y los ultimos datos de nuestra Data
print(DataFrameTitanic.info)

