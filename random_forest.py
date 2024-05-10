#%%
import pandas as pd
import rasterio
from rasterio import mask
import rasterio.plot as rio_plot
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from correlaciones_indices import df_indices
from correlaciones_bandas import df_bandas
from correlaciones_indices import df_indices
from correlaciones_sar_procesados import df_sar_noce

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

#%% hay que dividir los datos en características (X) y etiquetas/clasificación (y)
X = df.iloc[:, 0:108]
y = df["Clasif_1"]

#%% se separan los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.3, 
                                                    random_state=42)

#%% se debe crear el clasificador de Random Forest
rf_classifier = RandomForestClassifier(n_estimators=500, random_state=42)

#%% se entrena el clasificador con los datos de entrenamiento
rf_classifier.fit(X_train, y_train)
#%% predicciones sobre los datos de prueba
y_pred = rf_classifier.predict(X_test)

#%% matriz de confusión
confusion = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:")
print(confusion)

class_names = sorted(df["Clasif"].unique())
confusion_df = pd.DataFrame(confusion, index=class_names, columns=class_names)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(confusion_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
ax.set_xlabel("Predicción")
ax.set_ylabel("Etiqueta Verdadera")
ax.set_title("Matriz de Confusión")
plt.show()

#%% índice kappa
kappa = cohen_kappa_score(y_test, y_pred)
print("Índice kappa:", kappa)

#%% si se desea obtener la clasificación para todo el conjunto de datos original
y_pred_all = rf_classifier.predict(X)
df["Prediccion"] = y_pred_all

#%% guardar el df con las predicciones en un CSV
df.to_csv("C://Users//camil//Downloads//resultados.csv", index=False)

#%% VISUALIZACIÓN DEL MAPA CLASIFICADO

clasificado_path = "C://Users//camil//Downloads//resultados.csv"

# se carga como geodataframe
clasificado_gdf = gpd.read_file(clasificado_path)

# se carga el óptico
optico_s2_path = "C://Users//camil//Downloads//Salinidad//imagenes SALINIDAD//Sentinel2//Sentinel2_5abr2023.tif"
with rasterio.open(optico_s2_path) as src:
    optico_s2 = src.read(1)

# se obtienen los valores únicos de clasificación (no sé si usar Predicción o Clasif_1)
clases = sorted(clasificado_gdf["Clasif_1"].unique())

# paleta de colores
cmap = plt.cm.get_cmap("RdYlGn_r", len(clases))

# se crea una figura y un eje
fig, ax = plt.subplots(figsize=(10, 10))

# mostrar el mapa clasificado como una imagen con la paleta de colores
img = ax.imshow(optico_s2, cmap=cmap)

# se añade una barra de colores basada en la paleta de colores
cbar = plt.colorbar(img, ax=ax, ticks=np.arange(len(clases)))

# etiquetas de la barra de colores
cbar.set_ticklabels(clases)

# título
ax.set_title("Salinidad del suelo - random forest")

# visualización final
plt.show()

#%%
"""
###########################################################################
############### BORUTA ####################################################
###########################################################################
"""
"""
#%%
from boruta import BorutaPy
from sklearn.datasets import make_classification

#%%
# Genera un conjunto de datos de ejemplo
X, y = make_classification(n_samples=100, n_features=10, n_informative=3, n_classes=2, random_state=1)

# Inicializa el clasificador (puede ser cualquier clasificador que desees utilizar)
clf = RandomForestClassifier(n_estimators=100, random_state=1)

# Inicializa el algoritmo Boruta
boruta_selector = BorutaPy(clf, n_estimators='auto', verbose=2, random_state=1)

# Realiza la selección de características
boruta_selector.fit(X, y)

# Imprime las características seleccionadas
selected_features = X.columns[boruta_selector.support_].to_list()
print(selected_features)
"""
