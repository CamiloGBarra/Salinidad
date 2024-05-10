#%%
# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd
import statsmodels.api as sm
import geopandas as gpd
import seaborn as sns
import rasterio as rio
import rasterio.mask
from rasterstats import zonal_stats
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, GridSearchCV, ParameterGrid
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import glob
import os
import multiprocessing

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt

# Preprocesado y modelado
# ==============================================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.inspection import permutation_importance
import multiprocessing

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('once')

#%%
"""
Cargar ÍNDICES POLARIMÉTRICOS
"""
# el siguiente script lo que hace es extraer la información de los rasters ubicados en la carpeta de imágenes radar
# a partir de los polígonos creados en QGis (eran puntos que luego convertí a buffer). Crea la estadística y luego
# separa únicamente los valores medios. 

#%%
shapefile_path = r"C:\Users\camil\Downloads\Salinidad\python_salinidad\shapes\cobertura_de_suelo\buffer_cobertura_de_suelo1.shp"
poligonos = gpd.read_file(shapefile_path)

carpeta_imagenes = r"C:\Users\camil\Downloads\Salinidad\imagenes SALINIDAD\procesadas\PP\EOL1ASARSAO1A6877636_2023_04_06_orbit_DESCENDING\con speckle_refinedlee"
imagenes_satelitales = []

# Obtener la lista de archivos TIFF en la carpeta
archivos_tiff = glob.glob(os.path.join(carpeta_imagenes, "*.tif"))

# Iterar sobre cada archivo TIFF y cargarlo
for archivo_tiff in archivos_tiff:
    with rasterio.open(archivo_tiff) as src:
        imagen_satelital = src.read()
        imagenes_satelitales.append(imagen_satelital)

# Crear un diccionario para almacenar los datos de las estadísticas
datos_estadisticos = {
    'Poligono': [],
    'Clase': [],
    'ID': [],
    'Cobertura': []
}

# Iterar sobre cada imagen y calcular estadísticas para los polígonos
for i, imagen_satelital in enumerate(imagenes_satelitales):
    # Obtener la transformación de la imagen (para obtener coordenadas reales)
    transform = src.transform
    
    # Calcular estadísticas zonales para los polígonos
    stats = zonal_stats(poligonos, imagen_satelital[0], affine=transform, stats=['mean', 'min', 'max', 'sum'])
    
    # Procesar las estadísticas y agregarlas al diccionario
    for j, stat in enumerate(stats):
        datos_estadisticos[f'{os.path.basename(archivos_tiff[i])[:-4]}_min'] = datos_estadisticos.get(f'{os.path.basename(archivos_tiff[i])[:-4]}_min', []) + [stat['min']]
        datos_estadisticos[f'{os.path.basename(archivos_tiff[i])[:-4]}_max'] = datos_estadisticos.get(f'{os.path.basename(archivos_tiff[i])[:-4]}_max', []) + [stat['max']]
        datos_estadisticos[f'{os.path.basename(archivos_tiff[i])[:-4]}_mean'] = datos_estadisticos.get(f'{os.path.basename(archivos_tiff[i])[:-4]}_mean', []) + [stat['mean']]
        datos_estadisticos[f'{os.path.basename(archivos_tiff[i])[:-4]}_sum'] = datos_estadisticos.get(f'{os.path.basename(archivos_tiff[i])[:-4]}_sum', []) + [stat['sum']]
        
        if i == 0:
            datos_estadisticos['Poligono'].append(j+1)  # Numeración 1, 2, 3, ...
            datos_estadisticos['Clase'].append(poligonos['Clase'][j])  # Agregar la Clase del polígono
            datos_estadisticos['ID'].append(poligonos['ID'][j])  # Agregar el ID del polígono
            datos_estadisticos['Cobertura'].append(poligonos['cobertura'][j])  # Agregar la cobertura del polígono

# Crear el DataFrame a partir del diccionario
df_estadisticas = pd.DataFrame(datos_estadisticos)
    
# Seleccionar las columnas que contienen "mean" y la columna "Clase"
columnas_mean = df_estadisticas.filter(like='_mean').columns
columnas_seleccionadas = ['Clase'] + columnas_mean.tolist()

# Crear un nuevo DataFrame con las columnas seleccionadas
df = df_estadisticas[columnas_seleccionadas]

# :( Eliminar la parte "_mean" de los nombres de las columnas
nuevos_nombres_columnas = {col: col.replace('_mean', '') for col in df.columns if '_mean' in col}

# :) Renombrar las columnas
polarimetria = df.rename(columns=nuevos_nombres_columnas)

#%%
"""
Cargar ÍNDICES ÓPTICOS
"""

#%%
shapefile_path = r"C:\Users\camil\Downloads\Salinidad\python_salinidad\shapes\cobertura_de_suelo\buffer_cobertura_de_suelo1.shp"
poligonos = gpd.read_file(shapefile_path)

carpeta_imagenes = r"C:\Users\camil\Downloads\Salinidad\imagenes SALINIDAD\índices ópticos S2\5_abr"
imagenes_satelitales = []

# Obtener la lista de archivos TIFF en la carpeta
archivos_tiff = glob.glob(os.path.join(carpeta_imagenes, "*.tif"))

# Iterar sobre cada archivo TIFF y cargarlo
for archivo_tiff in archivos_tiff:
    with rasterio.open(archivo_tiff) as src:
        imagen_satelital = src.read()
        imagenes_satelitales.append(imagen_satelital)

# Crear un diccionario para almacenar los datos de las estadísticas
datos_estadisticos = {
    'Poligono': [],
    'Clase': [],
    'ID': [],
    'Cobertura': []
}

# Iterar sobre cada imagen y calcular estadísticas para los polígonos
for i, imagen_satelital in enumerate(imagenes_satelitales):
    # Obtener la transformación de la imagen (para obtener coordenadas reales)
    transform = src.transform
    
    # Calcular estadísticas zonales para los polígonos
    stats = zonal_stats(poligonos, imagen_satelital[0], affine=transform, stats=['mean', 'min', 'max', 'sum'])
    
    # Procesar las estadísticas y agregarlas al diccionario
    for j, stat in enumerate(stats):
        datos_estadisticos[f'{os.path.basename(archivos_tiff[i])[:-4]}_min'] = datos_estadisticos.get(f'{os.path.basename(archivos_tiff[i])[:-4]}_min', []) + [stat['min']]
        datos_estadisticos[f'{os.path.basename(archivos_tiff[i])[:-4]}_max'] = datos_estadisticos.get(f'{os.path.basename(archivos_tiff[i])[:-4]}_max', []) + [stat['max']]
        datos_estadisticos[f'{os.path.basename(archivos_tiff[i])[:-4]}_mean'] = datos_estadisticos.get(f'{os.path.basename(archivos_tiff[i])[:-4]}_mean', []) + [stat['mean']]
        datos_estadisticos[f'{os.path.basename(archivos_tiff[i])[:-4]}_sum'] = datos_estadisticos.get(f'{os.path.basename(archivos_tiff[i])[:-4]}_sum', []) + [stat['sum']]
        
        if i == 0:
            datos_estadisticos['Poligono'].append(j+1)  # Numeración 1, 2, 3, ...
            datos_estadisticos['Clase'].append(poligonos['Clase'][j])  # Agregar la Clase del polígono
            datos_estadisticos['ID'].append(poligonos['ID'][j])  # Agregar el ID del polígono
            datos_estadisticos['Cobertura'].append(poligonos['cobertura'][j])  # Agregar la cobertura del polígono

# Crear el DataFrame a partir del diccionario
df_estadisticas = pd.DataFrame(datos_estadisticos)
    
# Seleccionar las columnas que contienen "mean" y la columna "Clase"
columnas_mean = df_estadisticas.filter(like='_mean').columns
columnas_seleccionadas = ['Clase'] + columnas_mean.tolist()

# Crear un nuevo DataFrame con las columnas seleccionadas
df = df_estadisticas[columnas_seleccionadas]

# :( Eliminar la parte "_mean" de los nombres de las columnas
nuevos_nombres_columnas = {col: col.replace('_mean', '') for col in df.columns if '_mean' in col}

# :) Renombrar las columnas
opticos = df.rename(columns=nuevos_nombres_columnas)

#%%
#opticos = opticos[['Clase', 'SI1', 'BI','NDVI','MNDWI', 'NDI', 'SAVI']]

#polarimetria = polarimetria[['BET', 'ALP', "BMI", 'CSI', 'VSI', 'SPAM', 'PH', 'IRV', ]]
#polarimetria.drop(['GAM', 'DEL', 'fHHVV'], axis=1, inplace=True)

df = pd.concat([
#                    opticos, 
                    polarimetria
                    ], 
                    axis=1)
df = df.dropna()

X = df.drop(columns=["Clase"])
y = df["Clase"]

#%%
"""
opticos = r"C:\Users\camil\Downloads\Salinidad\python_salinidad\shapes\cobertura_de_suelo\opticos.csv"
opticos = pd.read_csv(opticos)
print("Columnas con valores NaN:", opticos.columns[opticos.isna().any()].tolist())
opticos = opticos[['SI1', 'BI','NDVI','MNDWI']]

polarimetria = r"C:\Users\camil\Downloads\Salinidad\python_salinidad\shapes\cobertura_de_suelo\buffer_estadisticas_polarimetria.csv"
polarimetria = pd.read_csv(polarimetria)
print("Columnas con valores NaN:", polarimetria.columns[polarimetria.isna().any()].tolist())
#polarimetria = polarimetria[['BET', 'ALP', "BMI", 'CSI', 'VSI', 'SPAM', 'PH', 'IRV', ]]
polarimetria.drop(['GAM', 'DEL', 'fHHVV'], axis=1, inplace=True)

dataset = pd.concat([
#                    opticos, 
                    polarimetria
                    ], 
                    axis=1)
#dataset = opticos
#dataset = polarimetria




coberturas = r"C:\Users\camil\Downloads\Salinidad\python_salinidad\shapes\cobertura_de_suelo\cobertura_de_suelo1.shp"
coberturas = gpd.read_file(coberturas)


df = pd.concat([coberturas, dataset], axis=1)
df = df.dropna()
df = df.drop(columns=["cobertura", "X", "Y", "geometry", "ID",
                      #"CRSI"
                      ])



X = df.drop(columns=["id"])
y = df["id"]

"""

#%% 
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=123)

#%%
# Grid de hiperparámetros evaluados
# ==============================================================================
param_grid = ParameterGrid(
                {'n_estimators': [100, 150, 200],
                 'max_features': [5, 7, 10],
                 'max_depth'   : [None, 3, 10, 20],
                 'criterion'   : ['gini', 'entropy']
                }
            )

# Loop para ajustar un modelo con cada combinación de hiperparámetros
# ==============================================================================
resultados = {'params': [], 'oob_accuracy': []}

for params in param_grid:
    
    modelo = RandomForestClassifier(
                oob_score    = True,
                n_jobs       = -1,
                random_state = 123,
                ** params
             )
    
    modelo.fit(X_train, y_train)
    
    resultados['params'].append(params)
    resultados['oob_accuracy'].append(modelo.oob_score_)
    print(f"Modelo: {params} \u2713")

# Resultados
# ==============================================================================
resultados = pd.DataFrame(resultados)
resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
resultados = resultados.sort_values('oob_accuracy', ascending=False)
resultados = resultados.drop(columns = 'params')
resultados.head(4)

#%%
modelo = RandomForestClassifier(
                n_estimators = 100,
                max_depth    = None,
                max_features = 5,
                oob_score    = True,
                n_jobs       = -1,
                random_state = 123
             )

#%%
modelo.fit(X_train, y_train)

#%%
y_pred = modelo.predict(X_test)

#%%
confusion = confusion_matrix(
                y_test, 
                y_pred
            )

print("Matriz de confusión:")
print(confusion)

class_names = sorted(df["Clase"].unique())
confusion_df = pd.DataFrame(confusion, index=class_names, columns=class_names)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(confusion_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
ax.set_xlabel("Predicción")
ax.set_ylabel("Etiqueta Verdadera")
ax.set_title("Matriz de Confusión")
plt.show()

#Error de test del modelo inicial
predicciones = modelo.predict (X =X_test)

rmse = mean_squared_error(
    y_true = y_test,
    y_pred = predicciones,
    squared = False
   )
print(f"El error (RMSE) de test es: {rmse}")

accuracy = accuracy_score(
            y_test, 
            y_pred,
            normalize = True
           )
print(f"El accuracy de test es: {100 * accuracy} %")

# Calcular el R2
r2 = r2_score(y_test, predicciones)
print(f"El coeficiente de determinación (R2) es: {r2}")

#%%
"""
VISUALIZACIÓN
"""

#%%
lista_par_training = list(df.columns.values)
lista_par_training = [x for x in lista_par_training if not 'id' in x]
lista_par_training

#%%
#Path donde estan las imagenes de los parámetros
path_indices_opticos = r"C:\Users\camil\Downloads\fusion\optico_alineados"
path_indices_polarimetricos = r"C:\Users\camil\Downloads\Salinidad\imagenes SALINIDAD\procesadas\PP\EOL1ASARSAO1A6877636_2023_04_06_orbit_DESCENDING\con speckle_refinedlee"
path_descomposiciones = r"C:\Users\camil\Downloads\fusion\separacion_descomposiciones"
seleccion = r"C:\Users\camil\Downloads\Salinidad\imagenes SALINIDAD\índices ópticos S2\seleccion"

imagenes_files = glob.glob(os.path.join(path_indices_polarimetricos,
                                        "**", "*.tif"), 
                           recursive=True)
nombres_imagenes = [os.path.basename(imagen_file) for imagen_file in imagenes_files]
nombres_imagenes

#%%
lista_par_training
lista_parametros = lista_par_training.copy()
for i in range(len(lista_parametros)):
    print(i)
    print(lista_parametros[i])

#%%
imagenes_files = [par for par in imagenes_files if any(xs in par for xs in lista_parametros)]

print('Lista parametros training: \n')
print(*imagenes_files, sep='\n \n')   
print('\n')

#%%
for id, layer in enumerate(imagenes_files, start=1):
    print(id, layer)

#%%
folder_out = "C://Users//camil//Downloads//"
# Read metadata of first file
with rio.open(imagenes_files[0]) as src0:
    meta = src0.meta

meta['count'] = len(imagenes_files)
print(meta)  
output_stack = os.path.join(folder_out, 'Stack_RF.tif' )

print('Archivo stack: \n')
print(output_stack)
print('\n')
print('--------------------------------------------------------------------------')


# Read each layer and write it to stack
with rio.open(output_stack, 'w', **meta) as dst:
    for id, layer in enumerate(imagenes_files, start=1):
        with rio.open(layer) as src1:
            dst.write_band(id, src1.read(1))

#%%
with rio.open(output_stack) as src:
    metadatos = src.meta.copy()
    stack_v = src.read()
    img = src.read(1)

stack_v = np.moveaxis(stack_v, 0, -1)

# Take our full image, ignore the Fmask band, and reshape into long 2d array (nrow * ncol, nband) for classification
new_shape = (stack_v.shape[0] * stack_v.shape[1], stack_v.shape[2])

img_as_array = stack_v[:, :, :].reshape(new_shape)
print('Reshaped from {o} to {n}'.format(o=stack_v.shape,
                                        n=img_as_array.shape))

img_as_array = np.nan_to_num(img_as_array)        


# Now predict for each pixel
class_prediction_v = modelo.predict(img_as_array)

# Reshape our classification map

class_prediction_v = class_prediction_v.reshape(stack_v[:, :, 0].shape)

#%%
nodatavalue = -999
img1 = np.nan_to_num(img, nan = nodatavalue)        

class_prediction_v[np.where(img1 == nodatavalue )] = nodatavalue

metadatos['count'] = 1
metadatos['nodata'] = nodatavalue

#%%
with rio.open(folder_out + 'cobertura_de_suelo.tif', 'w', **meta) as outf:
    outf.write(class_prediction_v, 1)
    
#%%
class_prediction_v[np.argwhere(np.isnan(img))] = -999

















