#%%
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn import set_config
import multiprocessing
import warnings
warnings.filterwarnings('ignore')
from correlaciones_indices import df_indices
from correlaciones_bandas import df_bandas
from correlaciones_indices import df_indices
from correlaciones_sar_procesados import df_sar_noce

#%% 
"""
DATOS
"""
#%%
shapefile_path = "C://Users//camil//Downloads//Salinidad//python_salinidad//shapes//puntos_abr2023.shp"
#sentinel_path = "C://Users//camil//Downloads//python_salinidad//shapes//optico_s2.tif"
#ndvi_path = "C://Users//camil//Downloads//python_salinidad//shapes//ndvi.tif"
#ndwi_path = "C://Users//camil//Downloads//python_salinidad//shapes//ndwi.tif"
#si1_path = "C://Users//camil//Downloads//python_salinidad//shapes//si1.tif"

#%%
proximidad_acequias_path = "C://Users//camil//Downloads//Salinidad//prueba_boruta//Descomposiciones//QGis//proximidad//proximidad_acequias.csv"
proximidad_acequias = pd.read_csv(proximidad_acequias_path)
proximidad_acequias_columna = proximidad_acequias["proximidad"]
#%%
clasificacion_salinidad = {
    (0, 2): {"Clasificación": "No salino", "Clasif_1": 0},
    (2, 4): {"Clasificación": "Ligeramente salino", "Clasif_1": 1},
    (4, 8): {"Clasificación": "Salinidad media", "Clasif_1": 2},
    (8, 16): {"Clasificación": "Fuertemente salino", "Clasif_1": 3},
    (16, float("inf")): {"Clasificación": "Extremadamente salino", "Clasif_1": 4}
}

#%% cargar el shapefile
shapefile = gpd.read_file(shapefile_path)

#%% se crea la lista para almacenar los datos
data = []

#%% 
"""
for i, point in shapefile.iterrows():
    # se obtienen las coordenadas del punto
    x = point.geometry.x
    y = point.geometry.y

    # se extraen los valores de los rasters en el punto
    with rasterio.open(sentinel_path) as src:
        sentinel_values = next(src.sample([(x, y)]))
    with rasterio.open(ndvi_path) as src:
        ndvi_value = next(src.sample([(x, y)]))
    with rasterio.open(ndwi_path) as src:
        ndwi_value = next(src.sample([(x, y)]))
    with rasterio.open(si1_path) as src:
        si1_value = next(src.sample([(x, y)]))

    # se agregan los valores a la lista
    row = [
        point["CE.2"],
        point["RAS"],
        ndvi_value[0],
        ndwi_value[0],
        si1_value[0],
        sentinel_values[0],
        sentinel_values[1],
        sentinel_values[2],
        sentinel_values[3],
        sentinel_values[4],
        sentinel_values[5],
        sentinel_values[6],
        sentinel_values[7],
        sentinel_values[8],
        sentinel_values[9],
        sentinel_values[10],
        sentinel_values[11],
        sentinel_values[12],
        #sentinel_values[13],
        #sentinel_values[14],
        #sentinel_values[15]
    ]
    data.append(row)
"""
#%% se crea el df
"""column_names = [
    "CE",
    "RAS",
    "NDVI",
    "NDWI",
    "SI1",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B9",
    "B10",
    "B11",
    "B12",
    "B13",
    #"B14",
    #"B15",
    #"B16"
]
df = pd.DataFrame(data, columns=column_names)
"""
#%%
df_bandas = df_bandas.drop(['CE'], axis=1)
df = pd.concat([df_bandas, df_indices, df_sar_noce], axis=1)
df.columns = ['CE' if col == 'CE' else col for col in df.columns]

#df = df_indices
#columnas_deseadas = ["CE.2", "NDSI", "NDWI", "SI1", "NDVI"]
#df = df_indices[columnas_deseadas]
#df = df_indices.drop(["SAI1", "SAI3", "SAI4", "SAI5", "BI", "TBI", "EVI", "SI3"], axis=1)
df = df.join(proximidad_acequias_columna)

#%% se agregan las columnas de clasificación
df["Clasif"] = df["CE"].apply(lambda x: next((v["Clasificación"] for k, v in clasificacion_salinidad.items() if k[0] <= x <= k[1]), ""))
df["Clasif_1"] = df["Clasif"].map({v["Clasificación"]: v["Clasif_1"] for k, v in clasificacion_salinidad.items()})

#%%
print(df)
df = df.dropna()

#%%
"""
Análisis exploratorio
"""
#%%
#Tipo de cada columna
#en pandas, el tipo 'object' hace referencia a strings
df.info()
#todas las columnas tienen el tipo de dato correcto

#número de datos ausentes por variable
df.isna().sum().sort_values()
#todas las columnas están completas, no hay valores ausentes

#distribución de la variable respuesta
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 3))
sns.histplot(data=df, x="CE", kde=True, ax=ax)
ax.set_title("Distribución salinidad")
ax.set_xlabel("CE");
#los modelos de redes neuronales son de tipo no paramétrico, no asumen ningún tipo de distribución de la variable
#respuesta, por lo tanto, no es necesario que esta siga ninguna distribución concreta.

"""#gráfico de distribución para cada variable numérica
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 7))
axes = axes.flat
columnas_numeric = datos.select_dtypes(include=['float64', 'int']).columns
columnas_numeric = columnas_numeric.drop('precio')

for i, colum in enumerate(columnas_numeric):
    sns.histplot(
        data    = datos,
        x       = colum,
        stat    = "count",
        kde     = True,
        color   = (list(plt.rcParams['axes.prop_cycle'])*2)[i]["color"],
        line_kws= {'linewidth': 2},
        alpha   = 0.3,
        ax      = axes[i]
    )
    axes[i].set_title(colum, fontsize = 7, fontweight = "bold")
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
    
    
fig.tight_layout()
plt.subplots_adjust(top = 0.9)
fig.suptitle('Distribución variables numéricas', fontsize = 10, fontweight = "bold");
"""
#%%
"""
División train y test
"""
#%%
#con el objetivo de poder estimar el error que comete el modelo al predecir nuevas observaciones
#se dividen los datos en dos grupos, uno de entrenamiento y otro de test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
                                        df.drop('CE', axis='columns'),
                                        df['CE'],
                                        train_size= 0.8,
                                        random_state= 42,
                                        shuffle = True #si se mezclan aleatoriamente las observaciones en cada iteración. Por defecto es True
                                    )
#tras realizar el reparto se verifica que los dos grupos son similares
print("Partición de entrenamiento")
print("--------------------------")
display(y_train.describe())
display(X_train.describe())
#display(X_train.describe(include='object'))
print("")

print("Partición de test")
print("--------------------------")
display(y_test.describe())
display(X_test.describe())
#display(X_test.describe(include='object'))

#%%
"""
Preprocesado (One-hot-encoding de las variables categóricas y estandarización de las
              variables continuas)
"""
#%%
#selección de las variables por tipo
#se estandarizan las columnas numéricas y se hace one-hot-encoding de las columnas cualitativas.
#Para mantener las columnas a las que no se les aplica ninguna transformación se tiene que
#indicar remainder='passthrough'

#identificación de columnas numéricas y categóricas
numeric_cols = X_train.select_dtypes(include=['float64', 'int']).columns.to_list()
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.to_list()

#transformación para las variables numéricas
numeric_transformer = Pipeline(
                        steps=[('scaler', StandardScaler())]
                    )

#transformación para las variables categóricas
categorical_transformer = Pipeline(
                            steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))]
                        )

preprocessor = ColumnTransformer(
                    transformers=[
                        ('numeric', numeric_transformer, numeric_cols),
                        ('cat', categorical_transformer, cat_cols)
                    ],
                    remainder='passthrough'
                )


set_config(display='diagram')
preprocessor

set_config(display='text')

#se aprenden y aplican las transformaciones de preprocesado
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

#convertir el output en df y añadir el nombre de las columnas
encoded_cat = preprocessor.named_transformers_['cat']['onehot']\
              .get_feature_names(cat_cols)
labels = np.concatenate([numeric_cols, encoded_cat])
datos_train_prep = preprocessor.transform(X_train)
datos_train_prep = pd.DataFrame(datos_train_prep, columns=labels)
datos_train_prep.info()

"""
Si bien realizar el preprocesado de forma separada del entrenamiento es útil 
para explorar y confirmar que las transformaciones realizadas son las deseadas, 
en la práctica, es más adecuado asociarlo al propio proceso de entrenamiento. 
Esto puede hacerse fácilmente en los modelos de scikit-learn con los Pipeline.
"""
#%%
"""
MODELADO . Pipeline de preprocesado + modelado
"""
#%%
#Pipeline
# Identificación de columnas numéricas y categóricas
numeric_cols = X_train.select_dtypes(include=['float64', 'int']).columns.to_list()
#cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.to_list()


# Transformaciones para las variables numéricas
numeric_transformer = Pipeline(
                        steps=[('scaler', StandardScaler())]
                      )

# Transformaciones para las variables categóricas
#categorical_transformer = Pipeline(
#                            steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))]
#                          )

preprocessor = ColumnTransformer(
                    transformers=[
                        ('numeric', numeric_transformer, numeric_cols),
#                        ('cat', categorical_transformer, cat_cols)
                    ],
                    remainder='passthrough'
                )

# Se combinan los pasos de preprocesado y el modelo en un mismo pipeline
pipe = Pipeline([('preprocessing', preprocessor),
                 ('modelo', MLPRegressor(solver = 'lbfgs', max_iter= 1000))
                ]
               )

# Espacio de búsqueda de cada hiperparámetro
param_distributions = {
    'modelo__hidden_layer_sizes': [(10), (20), (10, 10)],
    'modelo__alpha': np.logspace(-3, 3, 10),
    'modelo__learning_rate_init': [0.001, 0.01],
}

# Búsqueda por validación cruzada
grid = RandomizedSearchCV(
        estimator  = pipe,
        param_distributions = param_distributions,
        n_iter     = 50,
        scoring    = 'neg_mean_squared_error',
        n_jobs     = multiprocessing.cpu_count() - 1,
        cv         = 5, 
        verbose    = 0,
        random_state = 123,
        return_train_score = True
       )

grid.fit(X = X_train, y = y_train)

# Resultados del grid
resultados = pd.DataFrame(grid.cv_results_)
resultados.filter(regex = '(param.*|mean_t|std_t)')\
    .drop(columns = 'params')\
    .sort_values('mean_test_score', ascending = False)\
    .head(10)
    
#%%
"""
Error de test
"""
#%%
#Aunque mediante los métodos de validación (Kfold, LeaveOneOut) se consiguen buenas 
#estimaciones del error que tiene un modelo al predecir nuevas observaciones, la 
#mejor forma de evaluar un modelo final es prediciendo un conjunto test, es decir, 
#un conjunto de observaciones que se ha mantenido al margen del proceso de entrenamiento 
#y optimización.

# Error de test
modelo_final = grid.best_estimator_
predicciones = modelo_final.predict(X = X_test)
rmse = mean_squared_error(
        y_true = y_test,
        y_pred = predicciones,
        squared = False
       )
print('Error de test (rmse): ', rmse)

#%%
"""
Conclusión
"""
#%%
#La combinación de hiperparámetros con la que se obtienen mejores resultados 
#acorde a las metricas de validación cruzada es:

modelo_final['modelo'].get_params()











