""" Manipulacion de datos perdidos: En las Datas que nos vayamos encontrando veremos algunos casos que en las columnas de estos existen datos vacios
esto se le nombra como 'Datos perdidos', en el caso de ML esto nos puede afectar a la hora de entrenar nuestro modelo ya que por culpa de estos
puede que nuestro modelo no tenga la precision para realiar buenas predicciones, por esto veremos como manipular estos datos vacios
Signos que nos denomina si tenemos datos perdidos:
- '?'
- 'N/A'
- '0' A veces este no puede ser un datos vacio
- ' '
- 'NaN' Este es el mas comun

Antes de comenzar quiero aclarar que este proyecto se basa (o copia mas bien) del curso de Machine Learning del canal de: AprendeIA con Ligdi Gonzalez, 
fuente: https://www.youtube.com/watch?v=_i-c80qYqbs&list=PLJjOveEiVE4BK9Vnnl99H2IlYGhmokn1V&index=4
Nota: VEAN EL VIDEO EXPLICA LOS CASOS QUE PUEDOMOS HACER PARA LA MANIPULACION DE ESTOS """

#Antes de comenzar importaremos el modulos de pandas
import pandas as pd

#Importamos NumPy para los valores NaN
import numpy as np

#Importamos nuestros CSV (Para este caso no usare una variable para la URL, lo declarare directamente)
DataFrameTitanic = pd.read_csv("IntroduccionML/Preprocesamiento/Data/DataFrameEdit.csv")

#Probamos que nuestra Data fue importado con exito viendo
print(DataFrameTitanic.head())

""" Comenzamos con eliminar los datos faltantes, donde: 
- Axis = 0, Eliminar filas 
- Axis = 1, Eliminar columnas 
 """

#Primero eliminaremos las filas que contengan datos perdidos
print(DataFrameTitanic.dropna(axis = 0))

#Pero podemos hacerlo con una columna especifica
print(DataFrameTitanic.dropna(subset = ['Cabina'], axis = 0))

#Ahora con las columnas con datos perdidos perdidas
print(DataFrameTitanic.dropna(axis = 1))

""" Ahora para que se guarde en nuestro Data debemo de agregar el 'inplace = True'
DataFrameTitanic.dropna(axis = 0, inplace = True), Para este caso no lo impemetamos """

""" Remplazar los datos, para este caso nos basaremos en si existen datos vacios en la columna de edad
si este es el caso lo que haremos sera remplazar los datos perdidos por la media de la edad """
#Primero calculamos la media de la columna
print(DataFrameTitanic['Edad'].mean()) #El valor es de 29.69, lo redondeamos a 30

#Vamos a remplazar los datos
print(DataFrameTitanic['Edad'].replace(np.nan, 30))



