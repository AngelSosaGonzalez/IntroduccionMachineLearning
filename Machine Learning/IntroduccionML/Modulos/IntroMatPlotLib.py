""" En este archivo veremos los basico sobre MATPLOTLIB, este modulo de Python sirve para el graficado de datos,
este modulo es muy util para mostrar los datos (o un DataFrame ya que sabemos como realizarlo) """

""" NOTA: En cursos que te encuentre sobre NumPy o Pandas (o en este caso MATPLOTLIB) al momento de importar los modulos te encontraras 
con la palabra reservada "as" esto es un alias, asi podemos invocar las funciones del modulo sin escribir todo el 
nombre por el momento para entender mejor el modulo le pondremos alias """

#Importamos MATPLOTLIB
import matplotlib.pyplot as plt

#Graficado sencillo
#Creamos una lista de datos A y B
DatosA = [1, 2, 3]
DatosB = [11, 22, 33]

""" Graficamos los datos.+
Te voy a explicar que es cada elemento en la funcion "plot()"
- Datos A y B: Es la lista de datos donde la lista A son los datos del eje X y el B son los del eje y 
- Color = 'red': Es el color que tendra nuestra grafica
- linewidth = Este pertenece a la anchura de la linea
- label: El nombre de la grafica (o mas bien de la linea) """
plt.plot(DatosA, DatosB, color = 'red', linewidth = 3, label = 'lineal')
#Legend introuce un recuadro en la grafica describiendo el dato
plt.legend()
#Mostramos la grafica ya hecha
plt.show()

#Graficos
#Grafico de linea
#Creamos una lista de datos A y B
DatosA = [1, 2, 3]
DatosB = [11, 22, 33]

#Configuramos los parametros de nuestra grafica
plt.plot(DatosA, DatosB, color = 'red', linewidth = 3, label = 'lineal')

#Definimos los parametros de la grafica
#Agregamos un titulo a la grafica
plt.title('Grafica de linea')
#Titulo de los ejes
plt.ylabel('Eje Y')
plt.xlabel('Eje X')
#Agregamos una leyenda de informacion
plt.legend()
#Agregamos una cuadricula
plt.grid()
#Por ultimo mostramos la grafica
plt.show()

#Graficos de barras
#Creamos una lista de datos A y B
DatosA = [1, 2, 3]
DatosB = [11, 22, 33]

#Configuramos los parametros de nuestra grafica, pero en vez de usar "plot" usamos "bar"
plt.bar(DatosB, DatosA, color = 'red', width = 2, label = 'Barra')

#Definimos los parametros de la grafica
#Agregamos un titulo a la grafica
plt.title('Grafica de linea')
#Titulo de los ejes
plt.ylabel('Eje Y')
plt.xlabel('Eje X')
#Agregamos una leyenda de informacion
plt.legend()
#Por ultimo mostramos la grafica
plt.show()

#Histogramas
#Creamos una lista de datos A y B
DatosA = [11, 22, 33, 44, 55]
DatosB = [0, 10, 20, 30, 40, 50, 60]

#Configuramos los parametros de nuestra grafica, pero en vez de usar "plot" usamos "hist"
plt.hist(DatosB, DatosA, histtype='bar', rwidth = 0.5, color = 'red')

#Definimos los parametros de la grafica
#Agregamos un titulo a la grafica
plt.title('Histograma')
#Titulo de los ejes
plt.ylabel('Eje Y')
plt.xlabel('Eje X')
#Por ultimo mostramos la grafica
plt.show()

#Graficos de dispercion
#Creamos una lista de datos
DatosX1 = [0.5, 1.2, 2.7, 3.1, 4.2] 
DatosX2 = [0, 10, 20, 30, 40]

#Configuramos los parametros de nuestra grafica, pero en vez de usar "plot" usamos "scatter"
plt.scatter(DatosX1, DatosX2, label = 'Dispersion', color = 'red')

#Definimos los parametros de la grafica
#Agregamos un titulo a la grafica
plt.title('Grafica de dispersion')
#Titulo de los ejes
plt.ylabel('Eje Y')
plt.xlabel('Eje X')
#Agregamos una leyenda de informacion
plt.legend()
#Por ultimo mostramos la grafica
plt.show()

#Grafico circular
DatosX1 = [7, 8, 5, 11, 7]
DatosX2 = [2, 3, 4, 3, 2]

Divisiones = [2, 2]
#Ingresaremos el nombre que tendra nuestros datos
Datos = ['DatosA', 'DatosB']
#Definiremos los colores
Colores = ['red', 'blue']

#Configuramos los parametros de nuestra grafica, pero en vez de usar "plot" usamos "pie"
""" Definiremos los argumentos de la funcion:
- Divisiones = Es la porcion de cada dato 
- labels = El nombre de cada porcion
- startangle =  el angulo donde comenzara a partir la grafica
- shadow = Agregar sombra a nuestra grafica
- explode = Separar una porcion de la grafica para que destaque
- autopct = Los decimales o el valor que tomara (valga la redundancia) el valor del dato""" 
plt.pie(Divisiones, labels=Datos, colors=Colores, startangle=90, shadow=True, explode=(0,0), autopct='%1.1f%%')

#Introducir titulo
plt.title('Grafico de pastel')

#Mostramos el grafico
plt.show()