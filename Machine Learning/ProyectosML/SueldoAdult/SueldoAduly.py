""" Sueldo adultos: En este proyecto realizaremos en base a algoritmos de Machine Learning y visualizacion de los datos, saber
si la persona gana mas o menos de 50K, para esto vamos a tomar la Data que contiene informacion de adultos que en base a su 
informacion podremos determinar el resultaod deseado
Fuente: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/  """

#Importamos los modulos necesarios
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#Comenzamos con importar nuestros datos
DatosAdult = pd.read_csv('ProyectosML/SueldoAdult/Data/adult.csv')

#Crearemos las columnas necesarios para la data
DatosAdult.columns = ['Age', 'Workclass', 'Fnlwgt', 'Education', 'Education-num',
'Marital-statu', 'Occupation', 'Relationship', 'race', 'Sex', 'Capital-gain', 'Capital-loss',
'Hours-per-week', 'Native-country', 'Range']

#Ahora vamos a realzar el preprocesamiento
#Vemos si hay datos nulos
print(DatosAdult.isnull().sum())#Nos muestra que no hay datos nulos pero si observamos nuestra Data vemos que si hay pero en forma de '?'
""" Sabiendo esto vamos a sustituir los datos '?' por el valor '0', pero nos estamos adelantando, ahora vamos a realizar la 
sustitucion de los datos, verificamos los tipos de datos que tiene cada columna """
print(DatosAdult.info())
""" Datos a modificar:
- Workclass
- Education
- Marital-statu
- Occupation
- Relationship
- race
- Sex
- Native-country
- Range """
#*********************************************
DatosAdult['Workclass'].replace(
    (' ?', ' Self-emp-not-inc', ' Private', ' State-gov', ' Federal-gov', 
    ' Local-gov', ' Self-emp-inc', ' Without-pay', ' Never-worked'), 
    (0, 1, 2, 3, 4, 5, 6, 7, 8), 
    inplace=True)
#*********************************************
DatosAdult['Education'].replace(
    (' Bachelors', ' HS-grad', ' 11th', ' Masters', ' 9th', ' Some-college',
 ' Assoc-acdm', ' Assoc-voc', ' 7th-8th', ' Doctorate', ' Prof-school',
 ' 5th-6th', ' 10th', ' 1st-4th', ' Preschool', ' 12th'), 
 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), 
 inplace=True)
#**********************************************
DatosAdult['Marital-statu'].replace(
    (' Married-civ-spouse', ' Divorced', ' Married-spouse-absent',
 ' Never-married', ' Separated', ' Married-AF-spouse', ' Widowed'), 
 (0, 1, 2, 3, 4, 5, 6),
    inplace=True)
#*********************************************
DatosAdult['Occupation'].replace(
    (' ?' ,' Exec-managerial', ' Handlers-cleaners', ' Prof-specialty',
 ' Other-service', ' Adm-clerical', ' Sales', ' Craft-repair',
 ' Transport-moving', ' Farming-fishing', ' Machine-op-inspct',
 ' Tech-support', ' Protective-serv', ' Armed-Forces',
 ' Priv-house-serv'),
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
    inplace=True
)
#*********************************************
DatosAdult['Relationship'].replace(
    (' Husband', ' Not-in-family', ' Wife', ' Own-child', ' Unmarried',
 ' Other-relative'),
    (0, 1, 2, 3, 4, 5),
    inplace=True
)
#*********************************************
DatosAdult['race'].replace(
    (' White', ' Black', ' Asian-Pac-Islander', ' Amer-Indian-Eskimo', ' Other'),
    (0, 1, 2, 3, 4),
    inplace=True
)
#*********************************************
DatosAdult['Sex'].replace(
    (' Male', ' Female'),
    (0, 1),
    inplace=True
)
#*********************************************
Arreglo = DatosAdult['Native-country'].unique()

for i in range(Arreglo.size):
    if Arreglo[i] == ' ?':
        DatosAdult['Native-country'].replace((' ?'), (0), inplace=True)
    else: 
        DatosAdult['Native-country'].replace((Arreglo[i]), (i+1), inplace=True)
#*********************************************
DatosAdult['Range'].replace(
    (' <=50K', ' >50K'),
    (0, 1),
    inplace=True
)
#*********************************************
#Ya tenemos nuestra Data transformada por lo que vamos a realizar el uso de nuestro algortimo para realizar predicciones, separamos los datos
X = np.array(DatosAdult.drop(['Range'], 1))
Y = np.array(DatosAdult['Range'])

#Usaremos la funcion para seprara los datos en entrenamiento y prueba
X_Entre, X_Prueba, Y_Entre, Y_Prueba = train_test_split(X, Y, test_size = 0.2)

#Ahora vamos a invocar nuestro algoritmo
AlgoBosques = RandomForestClassifier(n_estimators= 200, max_depth=7)

#Entrenamos nuestro algoritmo
AlgoBosques.fit(X_Entre, Y_Entre)

#Calculamos nuestro score
print(AlgoBosques.score(X_Prueba, Y_Prueba))
#Nos da un porcentaje que ronda en los 85 a 86, es un buen numero para realizar predicciones

#Ahora en base al preprocesamiento de los datos vamos a realizar la visualizacion de datos, por lo que realizaremos sentecias
#1.- Porcentaje de rango de ganacion (mayor o menor 50K)
InfoDatos = DatosAdult['Range'].value_counts()
#Creamos una lista para el valor de los datos
Divicion = [InfoDatos[0], InfoDatos[1]]
#Ingresaremos el nombre que tendra nuestros datos
Datos = ['Mayor de 50K', 'Menor de 50K']
#Definiremos los colores
Colores = ['#3BCD69', '#CD563B']
#Configuramos los parametros de nuestra grafica, pero en vez de usar "plot" usamos "pie"
plt.pie(Divicion, labels=Datos, colors=Colores, startangle=90, shadow=True, explode=(0,0), autopct='%1.1f%%')
#Mostramos el grafico
plt.show()

""" En base a rangos de edad vermos cuales son los que estan entre los mayor de 50K los rangos son: 
De: 10 - 30
De: 30 - 60
De: 60 - 90 """
#Vamos a crear los rangos
Rango = [10, 30, 60, 90]
#Nombramos los rangos
RangoName = ['Joven', 'Adulto', 'Viejo']
#Aplicamos el cambio
DatosAdult['Age'] = pd.cut(DatosAdult['Age'], Rango, labels = RangoName)
#Realizaremos el conteo de los datos
Sentencia = pd.crosstab(index=DatosAdult['Range'], columns=DatosAdult['Age'])
Sentencia.index = ['Mayor de 50K', 'Menor de 50K']
print(Sentencia)

#Creamos un grupo por cada columna que hay
Joven = Sentencia['Joven']
Adulto = Sentencia['Adulto']
Viejo = Sentencia['Viejo']

#Ahora vamos a graficar
CantidadAlta = [Joven[0], Adulto[0], Viejo[0]]
CantidadBaja = [Joven[1], Adulto[1], Viejo[1]]
Labels = ['Joven', 'Adulto', 'Viejo']

#Que rango tiene mas de 50K
plt.bar(Labels, CantidadAlta)
plt.bar(Labels, CantidadBaja)
#Definimos los parametros de la grafica
#Agregamos un titulo a la grafica
plt.title('Edad y sueldo')
#Por ultimo mostramos la grafica
plt.show()