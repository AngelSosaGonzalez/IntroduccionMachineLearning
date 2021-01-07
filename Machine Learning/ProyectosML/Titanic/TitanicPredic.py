""" Prediccion Titanic: Ya conociendo todo lo que conlleva los temas introductorios de ML vamos a realizar nuestro primer proyecto practico
aqui vamos a aplicar todo el conocimiento que tuvimos de practica, para esto vamos a usar al Data de Titanic que lo encontraremos en la pagina 
de Kaggel, este es la mejor practica para comenzar asi que vamos.
Vamos a definir el problema, debemos de saber si los naufragos del titanic sobrevivieron o no, por lo que ya sabiendo esto nos damos cuenta
que necesitamos un algortmo de clasificacion donde nos clasifique si pertenece a un grupo o a otro.

Antes de comenzar quiero agradecer a 
- AprendeIA con Ligdi Gonzalez, su canal: https://www.youtube.com/channel/UCLJV54sFqPiH4MYcJKvGesg
- AMP Tech, su canal: https://www.youtube.com/channel/UCG4H4Qf-ZU9Ycr_PQ4egqDQ
Por los cursos de ML y Preprocesamiento de los datos, vean sus cursos son de demasiada ayuda """

#Este proyecto lo vamos a hacer paso por paso
#Primero veremos el preprocesamiento de los datos
#Importamos el modulos de Pandas y NumPy
import pandas as pd 
import numpy as np

#Ya teniendo nuestros modulos vamos a importar los datos (Entenamiento y prueba)
DatosTitanicEntre = pd.read_csv('ProyectosML/Titanic/Datos/train.csv') 
DatosTitanicPrue = pd.read_csv('ProyectosML/Titanic/Datos/test.csv')

#Comprovamos que se importaron bien
print(DatosTitanicEntre.head())
print(DatosTitanicPrue.head())

""" Viendo los datos (igual lo puedes ver directamente), en los datos de entrenamiento hay una columna nueva, donde te dice
si sobrevivio o no, esto ayudara para realizar la prediccion, antes de iniciar vamos ver los datos para ver si hay datos perdidos,
para ahorrarte esto la respuesta es SI """

#Vermos la informacio de nuestras datas
print(DatosTitanicEntre.info())
print(DatosTitanicPrue.info())
#Vemos que utilizan los mismos tipos de datos y vemos cuales datos son de tipo objeto y cuales no para asi cambiarlos de formato

#Ahora veremos si hay datos perdidos
print(DatosTitanicEntre.isnull().sum())
print(DatosTitanicPrue.isnull().sum())

""" Podemos llegar a la conclucion de que los datos perdidos son los de:
- 'Cabin'
- 'Age' 
Aqui podemos realizar:
- Eliminar columna o fila de datos perdidos
- Editar los datos perdidos
Aqui la cuestion es que si eliminamos las filas con los datos perdidos eliminaremos muchos datos valisos, pero tampoco podemos eliminar los
datos de edad porque es un factor a ver para la superviviencia, asi que lo que haremos es eliminar la columna cabina, ya que en los datos de
entrenamiento vemos que no le da mucha importancia y los datos de edad los vamos a sustituir """ 

#Eliminar la columna de Cabina o cabin (Datos faltantes)
DatosTitanicEntre.drop(['Cabin'], axis = 1, inplace = True)
DatosTitanicPrue.drop(['Cabin'], axis = 1, inplace = True)

#Ahora veremos si hay datos perdidos
print(DatosTitanicEntre.isnull().sum())
print(DatosTitanicPrue.isnull().sum())

#Ya eliminado vamos a realizar el cambio de la edad, para esto calculamos la media de estos
print(DatosTitanicEntre['Age'].mean())
print(DatosTitanicPrue['Age'].mean())
#Vemos que la media es de 29.6 y 30.27, por lo que lo vamos a redondear a 30

#Declaramos una variable con el dato de la edad que vamos a renplezar con los datos vacios
Edad = 30

#Vamos a remplazar los datos vacios o perdidos
DatosTitanicEntre['Age'].replace(np.nan, Edad, inplace = True)
DatosTitanicPrue['Age'].replace(np.nan, Edad, inplace = True)

#Ahora veremos si hay datos perdidos
print(DatosTitanicEntre.isnull().sum())
print(DatosTitanicPrue.isnull().sum())
#Vemos que ya no hay datos perdidos (Bueno algunos pero no le daremos importancia)

#Ahora vamos a convertir los datos objeto a numericos, como lo hemos visto esto es de mucha importancia para los algortimo
#Vermos la informacio de nuestras datas, para ver cuales son objetos
print(DatosTitanicEntre.info())
print(DatosTitanicPrue.info())

#Ahora veremos que datos hay que remplazar
print(DatosTitanicEntre.head())
print(DatosTitanicPrue.head())

""" Al ver los datos ponemos estos puntos:
- Sexo y Embarcacion es de tipo objeto y los vamos a cambiar a numerico
- Los datos como nombre y ticket son muy ambiguios por lo que al igual que la cabina los vamos a eliminar """

#Comenzamos con eliminar los nombres y los tickets
#Primero los de nombre
DatosTitanicEntre.drop(['Name'], axis = 1, inplace = True)
DatosTitanicPrue.drop(['Name'], axis = 1, inplace = True)

#Ahora los tickets
DatosTitanicEntre.drop(['Ticket'], axis = 1, inplace = True)
DatosTitanicPrue.drop(['Ticket'], axis = 1, inplace = True)

#Vemos si se realizaron los cambios
print(DatosTitanicEntre.head())
print(DatosTitanicPrue.head())

""" Vamos ahora con remplazar los datos del Sexo, para esto lo que haremos sera usar:
- 0 Para Mujeres 
- 1 Para Hombres """
DatosTitanicEntre['Sex'].replace(('male', 'female'), (1, 0), inplace = True)
DatosTitanicPrue['Sex'].replace(('male', 'female'), (1, 0), inplace = True)

#Vemos si se realizaron los cambios
print(DatosTitanicEntre.head())
print(DatosTitanicPrue.head())

""" Con los datos de embarcacion haremos lo mismo pero con:
- 0 Para S
- 1 Para C
- 2 Para Q """
DatosTitanicEntre['Embarked'].replace(('S', 'C', 'Q'),(0, 1, 2), inplace = True)
DatosTitanicPrue['Embarked'].replace(('S', 'C', 'Q'),(0, 1, 2), inplace = True)

""" ERROR: Input contains NaN, infinity or a value too large for dtype('float64').
Esto se debe a que tenemos datos perdidos aun no eliminados, estas son de embarcacion y fare
eliminamos las filas con dato perdidos """
DatosTitanicEntre.dropna(axis = 0, inplace = True)
DatosTitanicPrue.dropna(axis = 0, inplace = True)

#Ahora veremos si hay datos perdidos
print(DatosTitanicEntre.isnull().sum())
print(DatosTitanicPrue.isnull().sum())

#Vemos si se realizaron los cambios
print(DatosTitanicEntre.head())
print(DatosTitanicPrue.head())

#Por ultimo veremos la informacion de nuestra Data
#Tipos de datos
print(DatosTitanicEntre.info())
print(DatosTitanicPrue.info())

""" Ya viendo la informacion ya no hay datos perdidos ni de tipo objeto asi que vamos con el algoritmo, comunmente 
los modulos siempre se importan en el principio del codigo pero como estamos paso por paso vamos a importar los
modulos como vayamos avanzando, ahora para el uso de algoritmos de ML vamos a utilizar el modulo de Sklearn, y 
como al principio se intuyo el problema se necesita un algortmo de clasificacion por lo que usaremos distintos 
algoritmos de clasificacion:
- Vectores de soporte Clasificacion
- Regrecion Logistica
- Arboles 
- Bosque
Por obvias razones podemos utilizar menos algortimos, pero para practicar de esto vamos a utilizar los 4 y ver cual
algoritmo es mas preciso """

#Importamos los modulos de los algortmos
#Algortimo de vectores de soporte Clasificacion
from sklearn.svm import SVC

#Regrecion logistica
from sklearn.linear_model import LogisticRegression

#Arboles
from sklearn.tree import DecisionTreeClassifier

#Bosques
from sklearn.ensemble import RandomForestClassifier

#Separar los datos de entrenamiento y prueba
from sklearn.model_selection import train_test_split

""" Importamos metricas de rendimiento: 
- Matriz de confusión o error
- Precisión
- Recall o sensibilidad o TPR (Tasa positiva real)
- Exactitud
- Especificidad o TNR (Tasa negativa real)
- F1-Score """
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

#Consejo del canal de AprendeIA con Ligdi Gonzalez, separamos la columna de sobrevivientes de los datos de entrenamiento
#Este contiene los datos sin la columna de Survived
X = np.array(DatosTitanicEntre.drop(['Survived'], 1))
#Este contiene los datos con la columna Survived, para realizar las predicciones
Y = np.array(DatosTitanicEntre['Survived'])

#Separamos los datos en datos de entrenamiento y prueba
X_Entre, X_Prueba, Y_Entre, Y_Prueba = train_test_split(X, Y, test_size = 0.2)

#Ahora vamos a definir nuestros algoritmos
AlgoSVC = SVC(kernel='linear')
AlgoRegrLogic = LogisticRegression()
AlgoArbol = DecisionTreeClassifier(criterion='entropy')
AlgoBosques = RandomForestClassifier(n_estimators=10, criterion='entropy')

#Ya definido vamos a entrenarlos
AlgoSVC.fit(X_Entre, Y_Entre)
AlgoRegrLogic.fit(X_Entre, Y_Entre)
AlgoArbol.fit(X_Entre, Y_Entre)
AlgoBosques.fit(X_Entre, Y_Entre)

#Ahora calculamos el score
print(AlgoSVC.score(X_Prueba, Y_Prueba))
print(AlgoRegrLogic.score(X_Prueba, Y_Prueba))
print(AlgoArbol.score(X_Prueba, Y_Prueba))
print(AlgoBosques.score(X_Prueba, Y_Prueba))

#Ya entrenado vamos a realizar las predicciones
#Vamos a usar como referencia la ID de los naufragios
ID = DatosTitanicPrue['PassengerId']

#Ahora realizamos las predicciones SVC
PrediccionSVC = AlgoSVC.predict(DatosTitanicPrue)
#Creamos una DataFrame para ver las predicciones
DataPredicSVC = pd.DataFrame({'PassengerId': ID, 'Survived': PrediccionSVC})
#Imprimimos la prediccion
print(DataPredicSVC.head())

#Ahora realizamos las predicciones Regrecion logistica
PrediccionRegreLogi = AlgoRegrLogic.predict(DatosTitanicPrue)
#Creamos una DataFrame para ver las predicciones
DataPredicRegrelogi = pd.DataFrame({'PassengerId': ID, 'Survived': PrediccionRegreLogi})
#Imprimimos la prediccion
print(DataPredicRegrelogi.head())

#Ahora realizamos las predicciones Regrecion logistica
PrediccionArbol = AlgoArbol.predict(DatosTitanicPrue)
#Creamos una DataFrame para ver las predicciones
DataPredicArbol = pd.DataFrame({'PassengerId': ID, 'Survived': PrediccionArbol})
#Imprimimos la prediccion
print(DataPredicArbol.head())

#Ahora realizamos las predicciones Regrecion logistica
PrediccionBosque = AlgoBosques.predict(DatosTitanicPrue)
#Creamos una DataFrame para ver las predicciones
DataPredicBosques = pd.DataFrame({'PassengerId': ID, 'Survived': PrediccionBosque})
#Imprimimos la prediccion
print(DataPredicBosques.head())

#Ya hicimos nuestro primer proyecto de ML, felicidades este es el primer paso al camino de ser un experto