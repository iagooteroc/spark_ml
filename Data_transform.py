
# PRÁCTICA LIBRERÍAS - INTELIGENCIA COMPUTACIONAL PARA DATOS DE ALTA DIMENSIONALIDAD
# coding: utf-8

# In[1]:


from pyspark import SparkContext, SparkConf
from pyspark.mllib.util import MLUtils
import numpy as np
import time
from datetime import datetime



# In[2]:


SparkContext.setSystemProperty('spark.executor.memory', '4g')
conf = SparkConf().setAppName("MUBICS-IC-MLLIB").setMaster("local[2]")
sc = SparkContext(conf=conf)


# In[3]: Especificar ruta del conjunto de datos Global reef fish


FILE_NAME="Reef_Life_Survey_(RLS)#_Global_reef_fish_dataset-fish_surveys.csv"



# In[4]: TO-DO -> Leer el fichero en 64 particiones y en una sola línea
file = sc.textFile(FILE_NAME, minPartitions=64)


# In[5]: TO-DO -> Leer a una lista las columnas que aparezcan en la primera línea del fichero .csv
columnas = file.take(1)[0].split(',')

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

rddFilas = file.zipWithIndex().filter(lambda x: x[1]!=0).map(lambda x: arreglaComas(x[0]))

# cabecera = file.take(1)
# rddFilas = file.filter(lambda fila: fila!=cabecera).map(lambda fila: arreglaComas(fila))
print("TOTAL ROWS: %d"%rddFilas.count())



# In[7]: TO-DO -> Transformar el RDD de filas a las columnas que nos interesa: SurveyID, SiteLat, SiteLong, Family y Total
col_idx = []
col_idx.append(columnas.index("SurveyID"))
col_idx.append(columnas.index("SiteLat"))
col_idx.append(columnas.index("SiteLong"))
col_idx.append(columnas.index("Family"))
col_idx.append(columnas.index("Total"))

rddPreprocesado = rddFilas.map(lambda fila: fila.split(','))
rddProcesado = rddPreprocesado.map(lambda fila: list(filter(lambda x: x[0] in col_idx ,list(enumerate(fila))))).map(lambda x: [j for i,j in x])

print(rddProcesado.first())
# Esto queda como una lista de strings (un string por columna)

# In[8]: Los recuentos de una survey dada los transforma en un vector de recuentos por especies
# familias=map(lambda x: x.strip(),open("familias.txt",'r').readlines())
familias=open("familias.txt",'r').readlines()
familias = [x.replace("\n","") for x in familias] 
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
rddVectors=rddProcesado.map(lambda x: ((x[0],x[1],x[2]),(x[3],x[4])))\
	.groupByKey().map(lambda x: (x[0],toVector(x[1])))
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

dfVectors=rddVectors.map(lambda x: pasaFilaARow((x[0][1],x[0][2],x[1]))).toDF()
dfVectors.show()
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

# Análisis preliminar de datos
# Esquema:
dfVectors.printSchema()
# Nº de filas
print("Nº de filas: {}".format(dfVectors.count()))
# Estadísticas generales
dfVectors.describe().show()
# Nº de latitudes diferentes
dfVectors.select('latitude').distinct().count()
# Nº de longitudes diferentes
dfVectors.select('longitude').distinct().count()

# También pide número de registros, surveys, especies y familias
# Pero Procesado tiene un número diferente a Vectors :/
# dfProcesado = rddProcesado.toDF()
# dfProcesado.printSchema()
# dfProcesado.describe().show()


# In[10]: TO-DO -> Entrenamiento del modelo de Regresión a través del algoritmo Random forest regression de MLLib
# Partición del conjunto de datos: 70% Training - 30% Test

(trainingData, testData) = dfVectors.randomSplit([0.7,0.3])

for label in ["latitude", "longitude"]:

	# Train a RandomForest model.
	rf = RandomForestRegressor(featuresCol="features", labelCol=label)

	# Train model.
	model = rf.fit(trainingData)

	# Make predictions.
	predictions = model.transform(testData)

	# Select example rows to display.
	predictions.select("prediction", label, "features").show(5)

	# Select (prediction, true label) and compute test error
	evaluator = RegressionEvaluator(
	    labelCol=label, predictionCol="prediction", metricName="rmse")
	rmse = evaluator.evaluate(predictions)
	print("Root Mean Squared Error (RMSE) on %s test data = %g" % (label, rmse))
