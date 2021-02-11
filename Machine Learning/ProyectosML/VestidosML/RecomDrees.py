""" Recomendacion de vestidos: En este proyecto realizaremos en base a un algoritmo de Machine Learning predicciones para recomendar vestidos
dependiendo de sus caracteristicas, pero no solo eso tambien podremos algunas porblematicas para realizar graficas y mostrar el funcionamiento
de los datos """

#Importamos los modulos necesarios, pero esta vez vamos a separar por secciona para saber el funcionamiento de cada uno
#Modulos para importacion de la Data y la modificacion de esta
from joblib.logger import PrintTime
import pandas as pd
import numpy as np
#Modulos para realizar la separacion de datos y uso de algoritmo de ML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#Vamos a importar nuestra Data
DatosVestidos = pd.read_excel('Data/Attribute DataSet.xlsx', engine='openpyxl')
#NOTA: Para importar datos de Excel tendremos que requerir de openpyxl, se instala igual que cualquier modulo

#Vemos la informacio de nuestra data
print(DatosVestidos.info())
#Si hay datos vacios
print(DatosVestidos.isnull().sum())

""" Problema: Aqui en las ultimas 4 secciones podemos tenemos demaciados datos vacios, podemos realizar estas alternativas
- Borrar las columnas completas
- Borrar las filas completas
- Ver que dato se repite y llenarlo con esa media
Vamos a realiza la media ya que es la mitas de las columnas y filas, si realizamos alguna de las dos perderemos demaciada informacion """
#Antes de esto vamos a cambiar los datos de tipo objeto a numerico
#Estilo
Estilo = DatosVestidos['Style'].unique()
for i in range(Estilo.size):
    DatosVestidos['Style'].replace((Estilo[i]),(i),inplace=True)
#Precio
Precio = DatosVestidos['Price'].unique()
for i in range(Precio.size):
   if Precio[i] in ('Low', 'low'):
       DatosVestidos['Price'].replace((Precio[i]),(1),inplace=True)
   elif Precio[i] in ('High', 'high'):
       DatosVestidos['Price'].replace((Precio[i]),(4),inplace=True)

DatosVestidos['Price'].replace((np.nan, 'Average', 'Medium', 'very-high'),(0, 2, 3, 5), inplace=True)

#Talla
Talla = DatosVestidos['Size'].unique()
for i in range(Talla.size):
    if Talla[i] in ('S', 's', 'small'):
        DatosVestidos['Size'].replace((Talla[i]),(0),inplace=True)

DatosVestidos['Size'].replace(('M', 'L', 'XL', 'free'), (1, 2, 3, 4), inplace=True)

#Temporada
Temporada = DatosVestidos['Season'].unique()
for i in range(Temporada.size):
    if Temporada[i] in ('Summer', 'summer', np.nan):
        DatosVestidos['Season'].replace((Temporada[i]), (1), inplace=True)
    elif Temporada[i] in ('Spring','spring'):
        DatosVestidos['Season'].replace((Temporada[i]), (0), inplace=True) 
    elif Temporada[i] in ('Automn', 'Autumn'):
        DatosVestidos['Season'].replace((Temporada[i]), (2), inplace=True)
    elif Temporada[i] in ('Winter', 'winter'):
        DatosVestidos['Season'].replace((Temporada[i]), (3), inplace=True)

#Escote
Escote = DatosVestidos['NeckLine'].unique()
for i in range(Escote.size):
    if Escote[i] == np.nan:
        DatosVestidos['NeckLine'].replace((Escote[i]), (0), inplace=True)
    DatosVestidos['NeckLine'].replace((Escote[i]),(i), inplace=True)

#Manga
Manga = DatosVestidos['SleeveLength'].unique()
for i in range(Manga.size):
    if Manga[i] in ('sleevless', 'sleeveless', 'sleeevless', 'sleveless', np.nan):
        DatosVestidos['SleeveLength'].replace((Manga[i]), (0), inplace=True)
    if Manga[i] in ('threequarter',  'threequater', 'thressqatar'):
        DatosVestidos['SleeveLength'].replace((Manga[i]), (5), inplace=True)
    if Manga[i] in ('halfsleeve', 'half'):
        DatosVestidos['SleeveLength'].replace((Manga[i]), (6), inplace=True)
    
    DatosVestidos['SleeveLength'].replace((Manga[i]), (i), inplace=True)

DatosVestidos['SleeveLength'].replace((14), (9), inplace=True)#Esto porque cambia a la posicion que esta y queremos llevar un orden

#Material
Material = DatosVestidos['Material'].unique()
for i in range(Material.size):
    DatosVestidos['Material'].replace((Material[i]), (i), inplace=True)

for i in range(DatosVestidos['Material'].size):
    if i <= 250:
        DatosVestidos['Material'].replace((0), (4), inplace=True)
    elif i >= 250:
        DatosVestidos['Material'].replace((0), (1), inplace=True)

#Fabricacion
Fabrica = DatosVestidos['FabricType'].unique()
for i in range(Fabrica.size):
    if Fabrica[i] == np.nan:
        DatosVestidos['Material'].replace((np.nan), (1), inplace=True)
    DatosVestidos['FabricType'].replace((Fabrica[i]),(i),inplace=True)

#Waiseline
WaiseLine = DatosVestidos['waiseline'].unique()
for i in range(WaiseLine.size):
    if WaiseLine[i] == np.nan:
        DatosVestidos['waiseline'].replace((np.nan), (0), inplace=True)
    DatosVestidos['waiseline'].replace((WaiseLine[i]), (i), inplace=True)

#Decoracion
Decoracion = DatosVestidos['Decoration'].unique()
for i in range(Decoracion.size):
    if Decoracion[i] == np.nan:
        DatosVestidos['Decoration'].replace((np.nan), (0), inplace=True)
    DatosVestidos['Decoration'].replace((Decoracion[i]), (i), inplace=True)

#Patron
Patron = DatosVestidos['Pattern Type'].unique()
for i in range(Patron.size):
    if Patron[i] == np.nan:
        DatosVestidos['Pattern Type'].replace((np.nan), (1), inplace=True)
    DatosVestidos['Pattern Type'].replace((Patron[i]), (i), inplace=True)

#Rating, en esta parte como hay puntajes que utiliza decimales los vamos a agrupar
Rating = DatosVestidos['Rating'].unique()
for i in range(Rating.size):
    if Rating[i] > -1 and Rating[i] < 1:
        DatosVestidos['Rating'].replace((Rating[i]), (0), inplace=True)
    elif Rating[i] >= 1 and Rating[i] < 2:
        DatosVestidos['Rating'].replace((Rating[i]), (1), inplace=True)
    elif Rating[i] >= 2 and Rating[i] < 3:
        DatosVestidos['Rating'].replace((Rating[i]), (2), inplace=True)
    elif Rating[i] >= 3 and Rating[i] < 4:
        DatosVestidos['Rating'].replace((Rating[i]), (3), inplace=True)
    elif Rating[i] >= 4 and Rating[i] < 5:
        DatosVestidos['Rating'].replace((Rating[i]), (4), inplace=True)
    elif Rating[i] == 5:
        DatosVestidos['Rating'].replace((Rating[i]), (5), inplace=True)

print(DatosVestidos.isnull().sum())
print(DatosVestidos.info())

#Comenzamos con el algoritmo
#Separamos los datos en entrenamiento y prueba
X = np.array(DatosVestidos.drop(['Recommendation',], 1))
Y = np.array(DatosVestidos['Recommendation'])

X_Entre, X_Prueba, Y_Entre, Y_Prueba = train_test_split(X, Y, test_size = 0.1)

#Ahora seleccionamos nuestro algoritmo
Algoritmo = RandomForestClassifier(n_estimators=200, max_depth=30)
#Entrenamos nuestro algortimo
Algoritmo.fit(X_Entre, Y_Entre)

#Sacamos el score
print(Algoritmo.score(X_Prueba, Y_Prueba)) #Los numeros oscilan entre 60 y 70 es bueno, pero si realizamos predicciones en muy parecidos a los ultmos datos va a fallar

#De todos modos se pueden aumentar la data para realizar una prediccion acertada, aparte la data ya esta limpia de datos nulos y cambiadoa numericos

