#%%
# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from mlxtend.plotting import plot_decision_regions

# Preprocesado y modelado
# ==============================================================================
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

#%%
csv_muestreo_abril2023 = r"C:\Users\camil\Downloads\Salinidad\clasificacionTexttura_RF_GEE//suelo_abr2023.csv"
csv_muestreo_abril2023 = pd.read_csv(csv_muestreo_abril2023, sep=';', decimal='.')
#csv_muestreo_abril2023 = csv_muestreo_abril2023[['Muestra','CE', 'Humedad', 'RAS', 'pH', 'Suelo_desnudo_porc']]
Textura = csv_muestreo_abril2023[['Textura']]

#%%
opticos = r"C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\indices_opticos.csv"
opticos = pd.read_csv(opticos)
print("Columnas con valores NaN:", opticos.columns[opticos.isna().any()].tolist())
opticos = opticos[['BI', 'SI1', 'SI2', 'NDVI', 'KHAIER', 'NDSI', 'MNDWI']]

polarimetria = r"C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\indices_polarimetricos.csv"
polarimetria = pd.read_csv(polarimetria)
print("Columnas con valores NaN:", polarimetria.columns[polarimetria.isna().any()].tolist())
#polarimetria = polarimetria.iloc[:, 0:3]
polarimetria = polarimetria[['ENT', 'IRV', 'BET', 'ALP', 'GAM']]
polarimetria = polarimetria.iloc[:, 0:5]

dataset = pd.concat([opticos, polarimetria], axis=1)
dataset = opticos
dataset = polarimetria

#%%
clasificacion_textura = {
    'Franco': 0,
    'Franco arcillo limoso': 1,
    'Franco arcilloso': 2,
    'Franco limoso': 3
}

#%%
df = pd.concat([Textura, dataset], axis=1)

#%%
df["Clasif_1"] = df["Textura"].map(clasificacion_textura)

#%%
X = df.drop(columns=["Textura", "Clasif_1"])
y = df["Clasif_1"]

#%%
"""
Se ajusta primero un modelo SVM con kernel lineal y después uno con kernel radial, 
y se compara la capacidad de cada uno para clasificar correctamente las observaciones.
"""
#%%
"""
SVM LINEAL
"""

#%% 
X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        y.values.reshape(-1,1),
                                        train_size   = 0.75,
                                        random_state = 1234,
                                        shuffle      = True
                                    )

#%%
# Creación del modelo SVM lineal
# ==============================================================================
modelo = SVC(C = 100, kernel = 'linear', random_state=123)
modelo.fit(X_train, y_train)

#%%
# Predicciones test
# ==============================================================================
predicciones = modelo.predict(X_test)
predicciones

y_pred = modelo.predict(X_test)

#%%
# Accuracy de test del modelo 
# ==============================================================================
accuracy = accuracy_score(
            y_true    = y_test,
            y_pred    = predicciones,
            normalize = True
           )
print("")
print(f"El accuracy de test es: {100*accuracy}%")

# Matriz de confusión
# ==============================================================================

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

#%%
"""
SVM RADIAL
"""
#%%
# Grid de hiperparámetros
# ==============================================================================
param_grid = {'C': np.logspace(-5, 7, 20)}

# Búsqueda por validación cruzada
# ==============================================================================
grid = GridSearchCV(
        estimator  = SVC(kernel= "rbf", gamma='scale'),
        param_grid = param_grid,
        scoring    = 'accuracy',
        n_jobs     = -1,
        cv         = 3, 
        verbose    = 0,
        return_train_score = True
      )

# Se asigna el resultado a _ para que no se imprima por pantalla
_ = grid.fit(X = X_train, y = y_train)

# Resultados del grid
# ==============================================================================
resultados = pd.DataFrame(grid.cv_results_)
resultados.filter(regex = '(param.*|mean_t|std_t)')\
    .drop(columns = 'params')\
    .sort_values('mean_test_score', ascending = False) \
    .head(5)
    
#%%
# Mejores hiperparámetros por validación cruzada
# ==============================================================================
print("----------------------------------------")
print("Mejores hiperparámetros encontrados (cv)")
print("----------------------------------------")
print(grid.best_params_, ":", grid.best_score_, grid.scoring)

modelo = grid.best_estimator_

#%%
# Predicciones test
# ==============================================================================
predicciones = modelo.predict(X_test)
# Accuracy de test del modelo 
# ==============================================================================
accuracy = accuracy_score(
            y_true    = y_test,
            y_pred    = predicciones,
            normalize = True
           )
print("")
print(f"El accuracy de test es: {100*accuracy}%")

# Matriz de confusión de las predicciones de test
# ==============================================================================
confusion = confusion_matrix = pd.crosstab(
    y_test.ravel(),
    predicciones,
    rownames=['Real'],
    colnames=['Predicción']
)
confusion

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

imagenes_files = glob.glob(os.path.join(seleccion,
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
with rio.open(folder_out + 'Texturas_clasificacion_RF.tif', 'w', **metadatos) as outf:
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












