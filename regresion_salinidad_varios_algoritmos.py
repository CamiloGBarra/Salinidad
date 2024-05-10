#%%
# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd
from tabulate import tabulate
import glob
import os
import rasterio as rio

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.ticker as ticker
import seaborn as sns
import statsmodels.api as sm

# Preprocesado y modelado
# ==============================================================================
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedKFold, train_test_split
from sklearn.datasets import make_blobs
from sklearn.metrics import euclidean_distances, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence
import optuna

# ==============================================================================
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import make_column_selector

# Varios
# ==============================================================================
import multiprocessing
import random
from itertools import product
from fitter import Fitter, get_common_distributions

# Configuración matplotlib
# ==============================================================================
plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

#%% DATOS DE CAMPO

csv_muestreo_abril2023 = r"C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\suelo_abr2023.csv"
csv_muestreo_abril2023 = pd.read_csv(csv_muestreo_abril2023, sep=';', decimal='.')
#csv_muestreo_abril2023 = csv_muestreo_abril2023[['Muestra','CE', 'Humedad', 'RAS', 'pH', 'Suelo_desnudo_porc']]
CE = csv_muestreo_abril2023[['CE']]


# suelo desnudo SD >= 95
#SD_mayor_95 = csv_muestreo_abril2023.loc[csv_muestreo_abril2023['Suelo_desnudo_porc'] >= 95]
#CE = SD_mayor_95[['CE']]

#TRANSFORMADA
CE = np.sqrt(CE)


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

#%% Suelo desnudo (NDVI<0.25)

suelo_desnudo = r"C:\Users\camil\Downloads\PRUEBA PARA SUELO DESNUDO\suelo_desnudo.csv"
suelo_desnudo = pd.read_csv(suelo_desnudo)
#suelo_desnudo = suelo_desnudo.dropna()
suelo_desnudo.columns = [col.replace('Recorte', '') for col in suelo_desnudo.columns]
suelo_desnudo = suelo_desnudo.drop(["Muestra", "CE"], axis=1)

#%% REGIÓN 1 DEL PLOTEO H-ALPHA umbral alpha: 42 y entropía 0.5
regionA_ploteoHAlpha = r"C:\Users\camil\Downloads\Salinidad\Región A del ploteo H-Alpha\SAOCOM_region_A_ploteoHAlpha.csv"
regionA_ploteoHAlpha = pd.read_csv(regionA_ploteoHAlpha)
#regionA_ploteoHAlpha.columns = [col.replace('_mascara', '') for col in regionA_ploteoHAlpha.columns]
#regionA_ploteoHAlpha = regionA_ploteoHAlpha.drop(["ID"], axis=1)


#%% Acequias
proximidad_acequias = r"C:\Users\camil\Downloads\Salinidad\prueba_boruta\Descomposiciones\QGis\acequias\proximidad_acequias.csv"
proximidad_acequias = pd.read_csv(proximidad_acequias)
proximidad_acequias = proximidad_acequias[['proximidad_acequias']]
print("Columnas con valores NaN:", proximidad_acequias.columns[proximidad_acequias.isna().any()].tolist())

#%% índices ópticos y sar
opticos = r"C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\indices_opticos_5_abr.csv"
opticos = pd.read_csv(opticos)
print("Columnas con valores NaN:", opticos.columns[opticos.isna().any()].tolist())
opticos = opticos[[#'SI1', 
                   #'BI',
                   'NDVI',
                   'MNDWI',
                   'NDSI'
                   ]]

polarimetria = r"C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\indices_polarimetricos_buffer_medias_abr2023.csv"
polarimetria = pd.read_csv(polarimetria)
print("Columnas con valores NaN:", polarimetria.columns[polarimetria.isna().any()].tolist())
polarimetria = polarimetria.drop([#"PAU", 
                                  "FID"], axis=1)
#polarimetria = polarimetria.iloc[:, 0:18]
polarimetria = polarimetria[['ALP',
                             'ANY',
                             'BET',
                             'BMI',
                             'CSI',
                             'ENT',
                             'PAU',
                             'SPAM',
                             'VSI'
                             ]]

dataset = pd.concat([#opticos, 
                     polarimetria
                     ], 
                    axis=1)

#%%
"""
TODOS LOS DATOS
"""

#%%
df = pd.concat([CE, 
                dataset,
                #cobertura,
                #mapa_de_suelos,
                #proximidad_acequias,
                #suelo_desnudo,
                #regionA_ploteoHAlpha,
                ], axis=1)


#%%
"""
Análisis exploratorio previo
"""

#%%
df.head(4)

# Tipo de cada columna
# ==============================================================================
# En pandas, el tipo "object" hace referencia a strings
df.dtypes
df.info()

# Dimensiones del dataset
# ==============================================================================
df.shape

# Número de datos ausentes por variable
# ==============================================================================
df.isna().sum().sort_values

#df = df.dropna()


#%% Variable respuesta (CE)
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 6))
sns.kdeplot(
    df.CE,
    fill    = True,
    color   = "blue",
    ax      = axes[0]
)
sns.rugplot(
    df.CE,
    color   = "blue",
    ax      = axes[0]
)
axes[0].set_title("Distribución original", fontsize = 'medium')
axes[0].set_xlabel('CE', fontsize='small') 
axes[0].tick_params(labelsize = 6)

sns.kdeplot(
    np.sqrt(df.CE),
    fill    = True,
    color   = "blue",
    ax      = axes[1]
)
sns.rugplot(
    np.sqrt(df.CE),
    color   = "blue",
    ax      = axes[1]
)
axes[1].set_title("Transformación raíz cuadrada", fontsize = 'medium')
axes[1].set_xlabel('sqrt(CE)', fontsize='small') 
axes[1].tick_params(labelsize = 6)

sns.kdeplot(
    np.log(df.CE),
    fill    = True,
    color   = "blue",
    ax      = axes[2]
)
sns.rugplot(
    np.log(df.CE),
    color   = "blue",
    ax      = axes[2]
)
axes[2].set_title("Transformación logarítmica", fontsize = 'medium')
axes[2].set_xlabel('log(CE)', fontsize='small') 
axes[2].tick_params(labelsize = 6)

fig.tight_layout()

#%%
distribuciones = ['cauchy', 'chi2', 'expon',  'exponpow', 'gamma',
                  'norm', 'powerlaw', 'beta', 'logistic']

fitter = Fitter(df.CE, distributions=distribuciones)
fitter.fit()
fitter.summary(Nbest=10, plot=False)

#%%
# Correlación entre columnas numéricas
# ==============================================================================
def tidy_corr_matrix(corr_mat):
    '''
    Función para convertir una matrix de correlación de pandas en formato tidy
    '''
    corr_mat = corr_mat.stack().reset_index()
    corr_mat.columns = ['variable_1','variable_2','r']
    corr_mat = corr_mat.loc[corr_mat['variable_1'] != corr_mat['variable_2'], :]
    corr_mat['abs_r'] = np.abs(corr_mat['r'])
    corr_mat = corr_mat.sort_values('abs_r', ascending=False)
    
    return(corr_mat)

corr_matrix = df.select_dtypes(include=['float64', 'int']).corr(method='pearson')
tidy_corr_matrix(corr_matrix).head(10)

#%%
# Heatmap matriz de correlaciones
# ==============================================================================
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

sns.heatmap(
    corr_matrix,
    annot     = True,
    cbar      = False,
    annot_kws = {"size": 8},
    vmin      = -1,
    vmax      = 1,
    center    = 0,
    cmap      = sns.diverging_palette(20, 220, n=200),
    square    = True,
    ax        = ax
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation = 45,
    horizontalalignment = 'right',
)

ax.tick_params(labelsize = 8)

#%%
"""
Train y Test
"""
#%%
# Reparto de datos en train y test
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(
                                                    df.drop("CE", axis="columns"),
                                                    df['CE'],
                                                    train_size   = 0.7,
                                                    random_state = 123,
                                                    shuffle      = True
                                                    )

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

#%%
print("Partición de entrenamento")
print("-----------------------")
print(y_train.describe())

print("Partición de test")
print("-----------------------")
print(y_test.describe())

#%%
"""
K-Nearest Neighbor (kNN)
"""
#%%
# Modelo
modelo_knn = KNeighborsRegressor(n_neighbors = 6,
                             weights     = 'uniform',   #el peso de los vecinos. Uniforme o según la distancia
                             algorithm   = 'auto',
                             p           = 10,           #potencia para la métrica de distancia
                             n_jobs      = 1,
                             metric      = 'euclidean',
                             )

# Entrenamiento
modelo_knn.fit(X_train_scaled, y_train)

# Predicciones
y_pred = modelo_knn.predict(X_test_scaled)

# Calcular el MSE en el conjunto de prueba
mse_knn = mean_squared_error(
                            y_true  = y_test,
                            y_pred  = y_pred,
                            squared = False
                        )
print(f"El error cuadrático medio en el conjunto de prueba es: {mse_knn}") 


#%%
"""
Regresión lineal (Ridge y Lasso)
"""
#%%
# Modelo RIDGE
ridge_modelo = Ridge(alpha=5000)

# Entrenamiento
ridge_modelo.fit(X_train_scaled, y_train)

# predicciones
y_pred_ridge = ridge_modelo.predict(X_test_scaled)

# Calcular  el MSE en el conjunto de prueba
mse_ridge = mean_squared_error(
                            y_true  = y_test,
                            y_pred  = y_pred_ridge,
                            squared = False
                        )
print(f"El error cuadrático medio en el conjunto de prueba con RIDGE es: {mse_ridge}") 

#%%
# Modelo LASSO
lasso_modelo = Lasso(alpha=1000)

# Entrenamiento
lasso_modelo.fit(X_train_scaled, y_train)

# predicciones
y_pred_lasso = lasso_modelo.predict(X_test_scaled)

# Calcular  el MSE en el conjunto de prueba
mse_lasso = mean_squared_error(
                            y_true  = y_test,
                            y_pred  = y_pred_lasso,
                            squared = False
                        )
print(f"El error cuadrático medio en el conjunto de prueba con LASSO es: {mse_lasso}")

#%%
"""
Random forest
"""

#%%
# Modelo
rf_modelo = RandomForestRegressor(
                                    n_estimators = 150,
                                    criterion    = 'absolute_error',
                                    max_depth    = 3,
                                    max_features = None,
                                    oob_score    = False,
                                    n_jobs       = -1,
                                    random_state = 123
                                )

# Entrenamiento
rf_modelo.fit(X_train, y_train)

# Predicciones
y_pred_rf = rf_modelo.predict(X_test)

# Calcular el MSE
mse_rf = mean_squared_error(
                            y_true  = y_test,
                            y_pred  = y_pred_rf,
                            squared = False
                        )
print(f"MSE para Random Forest Regressor: {mse_rf}") 

r2_rf = r2_score(y_test, y_pred_rf)
print(f"R2 para Random Forest Regressor: {r2_rf}")
#%%
"""
Gradient Boosting Trees
"""

#%%
# Modelo
gb_modelo  = GradientBoostingRegressor(
                                        n_estimators = 200,
                                        learning_rate= 0.001,  
                                        loss         = 'absolute_error',
                                        max_features = 'sqrt',
                                        max_depth    = None,
                                        random_state = 123
                                    )

# Entrenamiento
gb_modelo.fit(X_train, y_train)

# Predicciones
y_pred_gb = gb_modelo.predict(X_test)

# Calcular MSE
mse_gb = mean_squared_error(
                            y_true  = y_test,
                            y_pred  = y_pred_gb,
                            squared = False
                        )
print(f"MSE para Gradient Boosting Regressor: {mse_gb}") 

r2_gb = r2_score(y_test, y_pred_gb)
print(f"R2 : {r2_gb}")

#%%
"""
Redes neuronales
"""

#%%
#Modelado
ann_regressor = MLPRegressor(
                        hidden_layer_sizes   = (8, 7), 
                        activation           = 'logistic',
                        solver               = 'sgd',
                        alpha                = 0.001,
                        max_iter             = 5000, 
                        random_state         = 123
                        )

ann_regressor.fit(X_train_scaled, y_train)

y_pred_ann = ann_regressor.predict(X_test_scaled)

mse_ann = mean_squared_error(
                            y_true  = y_test,
                            y_pred  = y_pred_ann,
                            squared = False
                        )
print(f"El error cuadrático medio con ANN es: {mse_ann}") 
r2_ann = r2_score(y_test, y_pred_ann)
print(f"R2: {r2_ann}")

#%%
"""
Support Vector Machine
"""

#%%
#Modelado
svm_regressor = SVR(kernel='linear')  # probar otros kernels como 'rbf' o 'poly'

svm_regressor.fit(X_train_scaled, y_train)

y_pred_svm = svm_regressor.predict(X_test_scaled)

mse_svm = mean_squared_error(
                            y_true  = y_test,
                            y_pred  = y_pred_svm,
                            squared = False
                        )
print(f'Mean Squared Error (MSE) con support vector machine: {mse_svm}')
r2 = r2_score(y_test, y_pred_svm)
print(f'R2 Score con support vector machine: {r2}')

#%%
"""
STAKING
"""
#%%
ridge_estimator = Ridge(alpha=1000)
lasso_estimator = Lasso(alpha=1000)
rf_estimator = RandomForestRegressor(
                                    n_estimators = 200,
                                    criterion    = 'absolute_error',
                                    max_depth    = 3,
                                    max_features = None,
                                    oob_score    = False,
                                    n_jobs       = -1,
                                    random_state = 123
                                )
gb_estimator = GradientBoostingRegressor(
                                        n_estimators        = 150, 
                                        random_state        = 123,
                                        validation_fraction = 0.001,
                                        n_iter_no_change    = 5,
                                        tol                 = 0.0001
                                    )

ann_regressor = MLPRegressor(
                        hidden_layer_sizes   = (20, 15), 
                        activation           = 'relu',
                        solver               = 'adam',
                        alpha                = 0.0001,
                        max_iter             = 1000, 
                        random_state         = 123
                        )

estimadores = [
                #('ridge', ridge_estimator),
                #('lasso', lasso_estimator),
                ('rf', rf_estimator), 
                ('gb', gb_estimator),
                ('ann', ann_regressor)
            ]

staking_modelo = StackingRegressor(estimators=estimadores, final_estimator=RidgeCV())

staking_modelo.fit(X_train, y_train)

y_pred_stacking = staking_modelo.predict(X_test)

mse_stacking = mean_squared_error(
                            y_true  = y_test,
                            y_pred  = y_pred_stacking,
                            squared = False
                        )
print(f"MSE para Stacking Regressor: {mse_stacking}") 

r2_stacking = r2_score(y_test, y_pred_stacking)
print(f'R2 : {r2_stacking}')

#%%
"""
Comparación
"""
#%%

error_modelos = pd.DataFrame({
                                'modelo': ['K-Nearest Neighbor (kNN)', 
                                           'Regresión lineal (Ridge)', 'Regresión lineal (Lasso)', 
                                           'Random Forest', 
                                           'Gradient Boosting Trees',
                                           'Staking (Super Learner)',
                                           'Redes neuronales',
                                           'Support Vector Machine'],
                                'rmse': [mse_knn, 
                                         mse_ridge, mse_lasso, 
                                         mse_rf, 
                                         mse_gb, 
                                         mse_stacking,
                                         mse_ann,
                                         mse_svm]
                        })

error_modelos = error_modelos.sort_values('rmse', ascending=False)

#%%
fig, ax = plt.subplots(figsize=(6, 3.84))
ax.hlines(error_modelos.modelo, xmin=0, xmax=error_modelos.rmse)
ax.plot(error_modelos.rmse, error_modelos.modelo, "o", color='black')
ax.tick_params(axis='y', which='major', labelsize=15)
ax.tick_params(axis='x', which='major', labelsize=15)
ax.set_title('Comparación de error de test modelos de regresión'),
ax.set_xlabel('Test rmse');

#%%
"""

Importancia de predictores

"""
#%% Importancia por pureza de nodos
importancia_predictores = pd.DataFrame(
                            {'predictor': df.drop(columns = "CE").columns,
                             'importancia': gb_modelo.feature_importances_}
                            )
print("Importancia de los predictores en el modelo")
print("-------------------------------------------")
importancia_predictores.sort_values('importancia', ascending=False)

#%% Importancia por permutación
from sklearn.inspection import permutation_importance

importancia = permutation_importance(
                estimator    = gb_modelo,
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
path_imagenes = r"C:\Users\camil\Downloads\regresion_salinidad"

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
folder_out = r"C:\Users\camil\Downloads\regresion_salinidad"
# Read metadata of first file
with rio.open(imagenes_files[0]) as src0:
    meta = src0.meta

meta['count'] = len(imagenes_files)
print(meta['count'])  
output_stack = os.path.join(folder_out, 'Stack.tif' )

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
class_prediction_v = gb_modelo.predict(img_as_array)

# Reshape our classification map

class_prediction_v = class_prediction_v.reshape(stack_v[:, :, 0].shape)

#%% 
nodatavalue = -999.0
img1 = np.nan_to_num(img, nan = nodatavalue)        

class_prediction_v[np.where(img1 == nodatavalue )] = nodatavalue

metadatos['count'] = 1
metadatos['nodata'] = nodatavalue

#%%
with rio.open(folder_out + 'Salinidad_regresion.tif', 'w', **meta) as outf:
    outf.write(class_prediction_v, 1)
    
            
#%%
class_prediction_v[np.argwhere(np.isnan(img))] = -999.0








