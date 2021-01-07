""" Importar y exportar datos: Esta practica veremos sobre como importar y exportar los datos con el fin de entender el procesos de prepocesado
de los datos, para este proyecto vamos a importar los datos de la pagina Kaggel, estos datos habla sobre los naufragos del TItanic donde veremos 
si estos viven o mueren (Lo se suena feo)
Antes de comenzar quiero aclarar que este proyecto se basa (o copia mas bien) del curso de Machine Learning del canal de: AprendeIA con Ligdi Gonzalez, 
fuente:https://www.youtube.com/watch?v=uaiYBm-ayio&list=PLJjOveEiVE4BK9Vnnl99H2IlYGhmokn1V&index=2 """

#Antes de iniciar vamos a importar los modulos necesarios
#Importaremos Pandas, esto para importar los datos de Kaggel
import pandas as pd #'as' es para darle un alias

""" Ya importado nuestro modulo, vamos a declarar una variable con la URL de la data, esto lo haremos para saber paso a paso sobre la exportacion de los datos,
aprte es para tener un orden en estos """
ImportURL = 'IntroduccionML/Preprocesamiento/Data/train.csv' #Para este caso usaremos datos locales, pero lo puedes hacer directo de la pagina solo copiando la URL

#Ahora con la URL declaradar vamos a importar nuestros datos
DataFrameTitanic = pd.read_csv(ImportURL) #'read_csv' es la funcion que nos ayuda a importar los datos
#Nota: Podemos ahorrarnos la declaracion de la variable poniendo la URL directo en la funcion, pero para conocer la funcion lo declaramos

#Visualizacion de datos
#Ver los primeros datos de la DataFrame
print(DataFrameTitanic.head())

#Visualizar los ultimos datos de la DataFrame
print(DataFrameTitanic.tail())

#Vamos a modificar las columnas de nuestro DataSet
Columnas = ['ID', 'Sobrevivio', 'Clase', 'Nombre', 'Sexo', 'Edad', 'Hermamos', 'Hijos', 'Boletos', 'Tarifa', 'Cabina', 'Embarcacion']

#Cambiamos los datos de la columna (Cambiamos el nombre)
DataFrameTitanic.columns = Columnas 

#Verificamos si se cambiaron los nombres de la columna
print(DataFrameTitanic.head()) 

""" Tenemos nuestro DataFrame editado, pero no guardado, por lo que vamos a guardar nuestro DataFRame editado, para esto haremos el mismo paso 
que la importacion del DataFrame pero usando otra funcion """
#Iniciamos declarando la variable de la URL (o direccion donde se guardara el documento)
ExportURL = 'IntroduccionML/Preprocesamiento/Data/DataFrameEdit.csv' #En la ruta para guardarlo lo pondremos nombrar como queramos
#Usaremos la funcion para guardar los datos 
DataFrameTitanic.to_csv(ExportURL)
