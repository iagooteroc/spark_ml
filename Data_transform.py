
# PRÁCTICA LIBRERÍAS - INTELIGENCIA COMPUTACIONAL PARA DATOS DE ALTA DIMENSIONALIDAD
# coding: utf-8

# In[1]:


from pyspark import SparkContext, SparkConf
from pyspark.mllib.util import MLUtils
import numpy as np
import time
from datetime import datetime


# In[2]:


SparkContext.setSystemProperty('spark.executor.memory', '8g')
conf = SparkConf().setAppName("MUBICS-IC-MLLIB").setMaster("local[4]")
sc = SparkContext(conf=conf)


# In[3]: Especificar ruta del conjunto de datos Global reef fish


FILE_NAME="Reef_Life_Survey_(RLS)#_Global_reef_fish_dataset-fish_surveys.csv"



# In[4]: TO-DO -> Leer el fichero en 64 particiones y en una sola línea



# In[5]: TO-DO -> Leer a una lista las columnas que aparezcan en la primera línea del fichero .csv


print(columnas)


# In[6]: TO-DO -> Contruir el RDD a partir del conjunto de datos, separando por columnas y eliminando la cabecera
# Consejo: Utilizar el método "arreglaComas", que codifica las comas que aparecen en cadenas literales y, por tanto, no separan campos. Tras utilizar el método, los campos se pueden obtener separando por ",".

def arreglaComas(x):
    nx=""
    numComillas=0
    for l in x:
        if l=='\"':
            numComillas+=1
        elif l==",":
            if numComillas%2==0:
                nx+=l
            else:
                nx+=";"
        else:
            nx+=l
    return nx


rddFilas=None #<- Completar
print("TOTAL ROWS: %d"%rddFilas.count())



# In[7]: TO-DO -> Transformar el RDD de filas a las columnas que nos interesa: SurveyID, SiteLat, SiteLong, Family y Total

rddProcesado=None #<- Completar
print(rddProcesado.first())


# In[8]: Los recuentos de una survey dada los transforma en un vector de recuentos por especies

familias=map(lambda x: x.strip(),open("familias.txt",'r').readlines())
def toVector(familyCountList):
    counts=np.zeros(len(familias))
    for (f,c) in familyCountList:
        if f in familias:
            counts[familias.index(f)]=c
    total=np.sum(counts)
    if total==0:
        return counts
    return counts/total


# TO-DO -> Agrupar las tuplas construidas anteriormente por localización (SurveryID, SiteLat, SiteLong) 
# Pista: Utilizar el método toVector


rddVectors=None #<- Completar
print(rddVectors.first())



# A partir de aquí se usará el API de DataFrames, por lo que es necesario pasar el RDD a DataFrame.

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("MUBICS-IC-MLLIB").getOrCreate()
        
from pyspark.sql.types import Row
from pyspark.ml.linalg import Vectors

# Dada una tupla (latitude, longitude, resultado de toVector del paso 8), esta función nos devuelve una row, de las que se compondrá un DataFrame.
def pasaFilaARow(x):
    d = {}
    d["latitude"]=float(x[0])
    d["longitude"]=float(x[1])
    d["features"]=Vectors.dense(x[2].tolist()+[float(x[0])])
    return Row(**d)


# TO-DO -> Conversión de RDD a DataFrame
# Pista: método toDF

dfVectors=None #<- Completar
dfVectors.show()



# In[10]: TO-DO -> Entrenamiento del modelo de Regresión a través del algoritmo Random forest regression de MLLib
# Partición del conjunto de datos: 70% Training - 30% Test


