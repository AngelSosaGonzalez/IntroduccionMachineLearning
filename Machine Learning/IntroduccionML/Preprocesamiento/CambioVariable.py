""" Cambio de variable cateforica a numerica: En la mayoria de los algoritmos de ML utiliza los datos numericos, en algunos casos mas no decir 
que la mayoria, siempre nos vamos a encontrar con datos de tipo 'object' (o cadena), esto nos puede dificultar las cosas al momentos de usar 
algoritmos de ML, por lo que vamos a ver como se cambian estas variables
Antes de comenzar quiero aclarar que este proyecto se basa (o copia mas bien) del curso de Machine Learning del canal de: AprendeIA con Ligdi Gonzalez, 
fuente: https://www.youtube.com/watch?v=Ij-j7XLVXCw&list=PLJjOveEiVE4BK9Vnnl99H2IlYGhmokn1V&index=5 """

#Importamos el modulos de Pandas
import pandas as pd

#Importamos nuestra Data
DataFrameTitanic = pd.read_csv('IntroduccionML/Preprocesamiento/Data/DataFrameEdit.csv')

#Verificamos si se importo correctamente
print(DataFrameTitanic.head())

""" Comenzamos con el cambio de variables, este caso veremos como este nos separa la columna en los datos object que aparece en la columna
tomamos como ejemplo la columna 'sexo', este tiene solo 2 datos: male, female, por lo que nos lo dividira en 2 columnas """
print(pd.get_dummies(DataFrameTitanic, columns=['Sexo']))

#Pero vemos que las columnas arrojan la misma informacion pero inversa por lo que podemos eliminar una de las comumnas
print(pd.get_dummies(DataFrameTitanic, columns=['Sexo'], drop_first=True))