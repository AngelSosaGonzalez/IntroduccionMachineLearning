""" Agrupacio de datos: El agrupamiento de datos o binning en ingles, es un método de preprocesamiento de datos y consiste en agrupar valores 
en compartimientos. En ocasiones este agrupamiento puede mejorar la precisión de los modelos predictivos y, a su vez, puede mejorar la 
comprensión de la distribución de los datos. 
Antes de comenzar quiero aclarar que este proyecto se basa (o copia mas bien) del curso de Machine Learning del canal de: AprendeIA con Ligdi Gonzalez, 
fuente: https://www.youtube.com/watch?v=Ij-j7XLVXCw&list=PLJjOveEiVE4BK9Vnnl99H2IlYGhmokn1V&index=5 """

#Comenzamos importando el modulo de Pandas
import pandas as pd 

#Importamos nuestros datos
DataFrameTitanic = pd.read_csv('IntroduccionML/Preprocesamiento/Data/DataFrameEdit.csv')

#Comprobamos si se importo correctamente
print(DataFrameTitanic.head())

""" Para este proyecto nos basaremos en los rangos dados por el tutorial que son: 
- El primer grupo lo comprenda las personas con edades entre 0 a 5,
- El segundo grupo serán las personas con edades entre 6 a 12,
- El tercer grupo estarán las personas entre 13 a 18 años,
- El cuarto grupo estará formado por las personas con edades comprendidas entre 19 a 35 años,
- El quinto lo forman las personas entre 36 años a 60, y
- El último grupo esta comprendido por las personas entre 61 año a 100 años. """

""" Comenzamos definiendo los rangos, este arreglo inicia con el numero donde inicia el primer rango '0' y 
los demas datos del arreglo con los numero que finalizan: '5, 12, 18, 35, 60, 100' """
Agrupacio = [0, 5, 12, 18, 35, 60, 100]

#Ahora nombramos nuestras agrupaciones, no nos rompemos la cabeza y solo le pondremos numeros de 1 - 6 (Puedes poner cualquier nombre)
Nombre = ['1', '2', '3', '4', '5', '6']

#Vamos a agrupar nuestro datos
print(pd.cut(DataFrameTitanic['Edad'], Agrupacio, labels = Nombre))




