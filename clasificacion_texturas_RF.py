#%%
import pandas as pd
import rasterio as rio
import rasterio.mask
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from sklearn.metrics import confusion_matrix, cohen_kappa_score, mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, GridSearchCV, ParameterGrid
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import glob
import os
import multiprocessing
from rasterstats import zonal_stats

#%%
shapefile_path = r"C:\Users\camil\Downloads\Salinidad\Texturas\buffer_texturas.shp"
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
    'Muestra': [],
    'Textura': []
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
            datos_estadisticos['Textura'].append(poligonos['Textura'][j])  # Agregar la Clase del polígono
            datos_estadisticos['Muestra'].append(poligonos['Muestra'][j])  # Agregar el ID del polígono
    
# Crear el DataFrame a partir del diccionario
df_estadisticas = pd.DataFrame(datos_estadisticos)
    
# Seleccionar las columnas que contienen "mean" y la columna "Clase"
columnas_mean = df_estadisticas.filter(like='_mean').columns
columnas_seleccionadas = ['Textura'] + columnas_mean.tolist()

# Crear un nuevo DataFrame con las columnas seleccionadas
df = df_estadisticas[columnas_seleccionadas]

# :( Eliminar la parte "_mean" de los nombres de las columnas
nuevos_nombres_columnas = {col: col.replace('_mean', '') for col in df.columns if '_mean' in col}

# :) Renombrar las columnas
polarimetria = df.rename(columns=nuevos_nombres_columnas)

#%%
#polarimetria = polarimetria[['BET', 'ALP', "BMI", 'CSI', 'VSI', 'SPAM', 'PH', 'IRV', ]]
#polarimetria.drop(['GAM', 'DEL', 'fHHVV'], axis=1, inplace=True)

#%%
texturas_counts = polarimetria['Textura'].value_counts()
plt.bar(texturas_counts.index, texturas_counts.values)
plt.ylabel('Frecuencia')
plt.title('Distribución de Texturas')
plt.xticks(rotation=90)
for i, v in enumerate(texturas_counts.values):
    plt.text(i, v + 0.5, str(v), ha='center')
plt.show()

#%%
clasificacion_textura = {
    'Franco': 0,
    'Franco arcillo limoso': 1,
    'Franco arcilloso': 2,
    'Franco limoso': 3
}

#%%
df["Clasif"] = df["Textura"].map(clasificacion_textura)

#%%
X = df.drop(columns=["Textura", "Clasif"])
y = df["Clasif"]

#%% 
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.3, 
                                                    random_state=123)

#%%
modelo = RandomForestClassifier(
                n_estimators = 500,
                max_depth    = None,
                max_features = 'sqrt',
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

class_names = sorted(df["Clasif_1"].unique())
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

#%%
"""
VISUALIZACIÓN
"""

#%%
lista_par_training = list(df.columns.values)
lista_par_training = [x for x in lista_par_training if not 'Textura' in x]
lista_par_training = [x for x in lista_par_training if not 'Clasif_1' in x]
lista_par_training

#%%
#Path donde estan las imagenes de los parámetros
path_indices_opticos = r"C:\Users\camil\Downloads\fusion\optico_alineados"
path_indices_polarimetricos = r"C:\Users\camil\Downloads\fusion\docker_alineados"
path_descomposiciones = r"C:\Users\camil\Downloads\fusion\separacion_descomposiciones"
seleccion = r"C:\Users\camil\Downloads\fusion\seleccion"

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

#%% Máscaras
#path_mascaras = r"C:\Users\camil\Downloads\Salinidad\Máscaras"
#mascaras = glob.glob(os.path.join(path_mascaras, "**", "*.tif"), recursive=True)
#nombres_mascaras = [os.path.basename(imagen_file) for imagen_file in mascaras] 
#nombres_mascaras
#path_out = r"C:\Users\camil\Downloads\Salinidad\clasificacionTexttura_RF_GEE"

# Ruta de los archivos Shapefile
ruta_planta_urbana = r"C:\Users\camil\Downloads\Salinidad\prueba_boruta\Descomposiciones\QGis\planta_urbana.shp"
ruta_rios_arroyos = r"C:\Users\camil\Downloads\Salinidad\prueba_boruta\Descomposiciones\QGis\IGN\rios_arroyos.shp"
ruta_red_vial = r"C:\Users\camil\Downloads\Salinidad\prueba_boruta\Descomposiciones\QGis\IGN\red_vial.shp"
ruta_lagos_lagunas = r"C:\Users\camil\Downloads\Salinidad\prueba_boruta\Descomposiciones\QGis\IGN\lagos_lagunas.shp"

# Cargar los Shapefiles en objetos GeoDataFrame
gdf_planta_urbana = gpd.read_file(ruta_planta_urbana)
gdf_rios_arroyos = gpd.read_file(ruta_rios_arroyos)
gdf_red_vial = gpd.read_file(ruta_red_vial)
gdf_lagos_lagunas = gpd.read_file(ruta_lagos_lagunas)

shapes = gpd.GeoDataFrame(pd.concat([gdf_planta_urbana, gdf_rios_arroyos, gdf_red_vial, gdf_lagos_lagunas], ignore_index=True))

#%%
nodatavalue = -999
img1 = np.nan_to_num(img, nan = nodatavalue)        

class_prediction_v[np.where(img1 == nodatavalue )] = nodatavalue

metadatos['count'] = 1
metadatos['nodata'] = nodatavalue

#%%
####################################### SIN MÁSCARAS #############################################
with rio.open(folder_out + 'Texturas_clasificacion_RF.tif', 'w', **meta) as outf:
    outf.write(class_prediction_v, 1)
    
#%%
####################################### CON MÁSCARAS #############################################
path_out = 'Texturas_clasificacion_RF.tif'

def write_tiff(path_out, np_array, shapes, meta):
    
    with rio.open(folder_out + path_out, "w", **metadatos) as dest:
        dest.write(np_array, 1)

    if shapes is not None:
        with rio.open(folder_out + path_out) as src:
            out_image, out_transform = rio.mask.mask(src, shapes, crop=False)

        with rio.open(folder_out + path_out, "w", **metadatos) as dest:
            dest.write(out_image)
            
write_tiff(path_out, class_prediction_v, None, **metadatos)
            
#%%
class_prediction_v[np.argwhere(np.isnan(img))] = -999












