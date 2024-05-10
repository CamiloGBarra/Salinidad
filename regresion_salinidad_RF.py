#%%
import pandas as pd
import rasterio as rio
import rasterio.mask
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import scipy.stats as stats
from sklearn.metrics import r2_score, cohen_kappa_score, mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, GridSearchCV, ParameterGrid
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import glob
import os
import multiprocessing
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import shapiro, boxcox

#%%
csv_muestreo_abril2023 = r"C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\suelo_abr2023.csv"
csv_muestreo_abril2023 = pd.read_csv(csv_muestreo_abril2023, sep=';', decimal='.')
#csv_muestreo_abril2023 = csv_muestreo_abril2023[['Muestra','CE', 'Humedad', 'RAS', 'pH', 'Suelo_desnudo_porc']]
CE = csv_muestreo_abril2023[['CE']]

#Transformadas
#CE = np.sqrt(CE)
CE = np.log(CE)

# suelo desnudo SD >= 95
#SD_mayor_95 = csv_muestreo_abril2023.loc[csv_muestreo_abril2023['Suelo_desnudo_porc'] >= 95]
#CE = SD_mayor_95[['CE']]

sns.histplot(CE, kde=True)
plt.show()

# Q-Q plot para evaluar la normalidad
stats.probplot(CE['CE'], dist="norm", plot=plt)
plt.show()

stat, p_value = shapiro(CE['CE'])
print(f'p-value: {p_value}')

#%% Mapa de suelos
mapa_de_suelos = r"C:\Users\camil\Downloads\Salinidad\python_salinidad\shapes\mapa_de_suelos.csv"
mapa_de_suelos = pd.read_csv(mapa_de_suelos)
mapa_de_suelos = mapa_de_suelos[['mapa_de_suelos_rasterizado']]
mapa_de_suelos.rename(columns={'mapa_de_suelos_rasterizado': 'mapa_de_suelos'}, inplace=True)
mapeo = {4: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8}
mapa_de_suelos['mapa_de_suelos'] = mapa_de_suelos['mapa_de_suelos'].replace(mapeo)
mapeo_serie = {
    1: 'Belgrano',
    2: 'Cortinez',
    3: 'Complejo fluvial',
    4: 'Canal Puntilla',
    5: 'Mitre',
    6: 'Sarmiento',
    7: 'Pie de palo',
    8: 'Roca-urbano'
}
mapa_de_suelos['serie'] = mapa_de_suelos['mapa_de_suelos'].map(mapeo_serie)

mapa_de_suelos = mapa_de_suelos[['mapa_de_suelos']]

#%% Cobertura de suelo
cobertura = r"C:\Users\camil\Downloads\Salinidad\python_salinidad\shapes\cobertura.csv"
cobertura = pd.read_csv(cobertura)
cobertura.rename(columns={'cobertura_de_suelo_classRF_1': 'cobertura'}, inplace=True)
mapeo1 = {1: 3} # hay uno solo de urbano, así que lo paso a suelo desnudo
cobertura['cobertura'] = cobertura['cobertura'].replace(mapeo1)
cobertura = cobertura[['cobertura']]

# 1 : urbano
# 2 : agua
# 3 : suelo desnudo
# 4 : vegetación

#%% Acequias
proximidad_acequias = r"C:\Users\camil\Downloads\Salinidad\prueba_boruta\Descomposiciones\QGis\acequias\proximidad_acequias.csv"
proximidad_acequias = pd.read_csv(proximidad_acequias)
proximidad_acequias = proximidad_acequias[['proximidad_acequias']]
print("Columnas con valores NaN:", proximidad_acequias.columns[proximidad_acequias.isna().any()].tolist())

#%% REGIÓN 1 DEL PLOTEO H-ALPHA umbral alpha: 42 y entropía 0.5
regionA_ploteoHAlpha = r"C:\Users\camil\Downloads\Salinidad\Región A del ploteo H-Alpha\SAOCOM\SAOCOM_region_A_ploteoHAlpha.csv"
regionA_ploteoHAlpha = pd.read_csv(regionA_ploteoHAlpha)
regionA_ploteoHAlpha.columns = [col.replace('_mascara', '') for col in regionA_ploteoHAlpha.columns]
regionA_ploteoHAlpha = regionA_ploteoHAlpha.drop(["ID"], axis=1)

regionA_ploteoHAlpha = regionA_ploteoHAlpha[["ANY"]]

#%% Con mútiples máscaras (región 1 del ploteo H-ALPHA, suelo desnudo, urbano, agua y red vial)

opticos = r"C:\Users\camil\Downloads\Salinidad\Multiples mascaras\Sentinel2.csv"
opticos = pd.read_csv(opticos)
print("Columnas con valores NaN:", opticos.columns[opticos.isna().any()].tolist())
opticos = opticos.drop(["Muestra"], axis=1)
#opticos = opticos[['SI1', 'BI','NDVI','MNDWI']]

polarimetria = r"C:\Users\camil\Downloads\Salinidad\Multiples mascaras\SAOCOM.csv"
polarimetria = pd.read_csv(polarimetria)
print("Columnas con valores NaN:", polarimetria.columns[polarimetria.isna().any()].tolist())
polarimetria = polarimetria.drop([#"PAU", 
                                  "Muestra"], axis=1)
#polarimetria = polarimetria.iloc[:, 0:18]
polarimetria = polarimetria[['BET', 'BMI', 'PAU', 'SPAM', 'ENT', 'ALP']]

dataset = pd.concat([#opticos, 
                     polarimetria
                     ], 
                    axis=1)

#%% INDICES ÓPTICOS Y SAR
opticos = r"C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\indices_opticos_5_abr.csv"
opticos = pd.read_csv(opticos)
print("Columnas con valores NaN:", opticos.columns[opticos.isna().any()].tolist())
opticos = opticos[[#'SI1', 
                   'BI',
                   'NDVI',
                   'MNDWI',
                   #'NDSI'
                   ]]

polarimetria = r"C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\indices_polarimetricos_buffer_medias_abr2023.csv"
polarimetria = pd.read_csv(polarimetria)
print("Columnas con valores NaN:", polarimetria.columns[polarimetria.isna().any()].tolist())
polarimetria = polarimetria.drop([#"PAU", 
                                  "FID"], axis=1)
#polarimetria = polarimetria.iloc[:, 0:18]
polarimetria = polarimetria[[#'ALP',
                             'ANY',
                             'BET',
                             #'BMI',
                             'CSI',
                             'ENT',
                             #'PAU',
                             #'SPAM',
                             'VSI',
                             'IRV',
                             #'mHHVV',
                             #'mHVHH',
                             #'mVHVV',
                             #'PH',
                             #'RVV',
                             #'RVH',
                             #'RHV',
                             #'RHH'
                             ]]

dataset = pd.concat([#opticos, 
                     polarimetria
                     ], 
                    axis=1)

#%%
df = pd.concat([
                CE, 
                #proximidad_acequias,
                #cobertura,
                #mapa_de_suelos,
                dataset,
                #suelo_desnudo,
                #regionA_ploteoHAlpha
                ], 
               axis=1)

df = df.dropna()

#%%
X = df.drop(columns=["CE"])
y = df["CE"]

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.30, 
                                                    random_state=123)

#%%
# Creación del modelo
# ==============================================================================
modelo = RandomForestRegressor(
            n_estimators = 80,
            criterion    = 'squared_error',
            max_depth    = 9,
            max_features = 1,
            oob_score    = False,
            n_jobs       = -1,
            random_state = 123
         )

# Entrenamiento
modelo.fit(X_train, y_train)

# Error de test del modelo inicial
# ==============================================================================
y_pred = modelo.predict(X = X_test)

rmse = mean_squared_error(
        y_true  = y_test,
        y_pred  = y_pred,
        squared = False
       )
print(f"El error (rmse) de test es: {rmse}")

r2_rf = r2_score(y_test, y_pred)
print(f"R2 para Random Forest Regressor: {r2_rf}")

#%% Gráfico de dispersión de las predicciones vs. los valores reales:
plt.scatter(y_test, y_pred)
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
plt.title("Gráfico de dispersión de Random Forest")
plt.show()

# Gráfico de distribución de residuos:
residuals = y_test - y_pred
plt.hist(residuals, bins=30)
plt.xlabel("Residuos")
plt.ylabel("Frecuencia")
plt.title("Distribución de Residuos")
plt.show()  
    
#%%
"""
Optimización de hiperparámetros
"""
#%%
# Grid de hiperparámetros evaluados
# ==============================================================================
param_grid = ParameterGrid(
                {'n_estimators': [100, 150, 200],
                 'max_features': [3, 5, 7, 9, 11],
                 'max_depth'   : [None, 3, 10, 20]
                }
             )

# Loop para ajustar un modelo con cada combinación de hiperparámetros
# ==============================================================================
resultados = {'params': [], 'oob_r2': []}

for params in param_grid:
    
    modelo = RandomForestRegressor(
                oob_score    = True,
                n_jobs       = -1,
                random_state = 123,
                ** params
             )
    
    modelo.fit(X_train, y_train)
    
    resultados['params'].append(params)
    resultados['oob_r2'].append(modelo.oob_score_)
    print(f"Modelo: {params} \u2713")

# Resultados
# ==============================================================================
resultados = pd.DataFrame(resultados)
resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
resultados = resultados.drop(columns = 'params')
resultados = resultados.sort_values('oob_r2', ascending=False)
resultados.head(4)

#%%
"""
Importancia de predictores
"""
#%% Importancia por pureza de nodos
importancia_predictores = pd.DataFrame(
                            {'predictor': df.drop(columns = "CE").columns,
                             'importancia': modelo.feature_importances_}
                            )
print("Importancia de los predictores en el modelo")
print("-------------------------------------------")
importancia_predictores.sort_values('importancia', ascending=False)

#%% Importancia por permutación
importancia = permutation_importance(
                estimator    = modelo,
                X            = X_train,
                y            = y_train,
                n_repeats    = 5,
                scoring      = 'neg_root_mean_squared_error',
                n_jobs       = multiprocessing.cpu_count() - 1,
                random_state = 123
             )

# Se almacenan los resultados (media y desviación) en un dataframe
df_importancia = pd.DataFrame(
                    {k: importancia[k] for k in ['importances_mean', 'importances_std']}
                 )
df_importancia['feature'] = X_train.columns
df_importancia.sort_values('importances_mean', ascending=False)

#%%
# Importancia por permutación= Gráfico
fig, ax = plt.subplots(figsize=(5, 6))
df_importancia = df_importancia.sort_values('importances_mean', ascending=True)
ax.barh(
    df_importancia['feature'],
    df_importancia['importances_mean'],
    xerr=df_importancia['importances_std'],
    align='center',
    alpha=0
)
ax.plot(
    df_importancia['importances_mean'],
    df_importancia['feature'],
    marker="D",
    linestyle="",
    alpha=0.8,
    color="r"
)
ax.set_title('Importancia de los predictores (train)')
ax.set_xlabel('Incremento del error tras la permutación');

#%%
"""
VISUALIZACIÓN
"""

#%%
#ordena las columnas por abecedario
df = df.sort_index(axis=1)

lista_par_training = list(df.columns.values)
lista_par_training = [x for x in lista_par_training if not 'CE' in x]
lista_par_training

#%%
#Path donde estan las imagenes de los parámetros
path_imagenes = r"C:\Users\camil\Downloads\PRUEBA PARA SUELO DESNUDO"

imagenes_files = glob.glob(os.path.join(path_imagenes,
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
folder_out = r"C:\Users\camil\Downloads\PRUEBA PARA SUELO DESNUDO"
# Read metadata of first file
with rio.open(imagenes_files[0]) as src0:
    meta = src0.meta

meta['count'] = len(imagenes_files)
print(meta['count'])  
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
nodatavalue = -999.0
img1 = np.nan_to_num(img, nan = nodatavalue)        

class_prediction_v[np.where(img1 == nodatavalue )] = nodatavalue

metadatos['count'] = 1
metadatos['nodata'] = nodatavalue

#%%
with rio.open(folder_out + 'Salinidad_regresion_RF.tif', 'w', **meta) as outf:
    outf.write(class_prediction_v, 1)
    
            
#%%
class_prediction_v[np.argwhere(np.isnan(img))] = -999.0


