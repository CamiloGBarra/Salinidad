#%%
# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt

# Preprocesado y modelado
# ==============================================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
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

#%%
csv_muestreo_abril2023 = r"C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\suelo_abr2023.csv"
csv_muestreo_abril2023 = pd.read_csv(csv_muestreo_abril2023, sep=';', decimal='.')
#csv_muestreo_abril2023 = csv_muestreo_abril2023[['Muestra','CE', 'Humedad', 'RAS', 'pH', 'Suelo_desnudo_porc']]
CE = csv_muestreo_abril2023[['CE']]
#CE = np.log(CE)

# suelo desnudo SD >= 95
#SD_mayor_95 = csv_muestreo_abril2023.loc[csv_muestreo_abril2023['Suelo_desnudo_porc'] >= 95]
#CE = SD_mayor_95[['CE']]

CE.hist(bins=50, figsize=(10, 6))
plt.suptitle('Distribución de Datos en DataFrame CE')
plt.show()

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(CE, bins=100, kde=True)
plt.title('Distribución de Datos en DataFrame CE')
plt.show()

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

#%% Suelo desnudo (NDVI<0.25)

suelo_desnudo = r"C:\Users\camil\Downloads\Salinidad\Suelo desnudo\suelo_desnudo.csv"
suelo_desnudo = pd.read_csv(suelo_desnudo)
#suelo_desnudo = suelo_desnudo.dropna()
suelo_desnudo.columns = [col.replace('Recorte', '') for col in suelo_desnudo.columns]
suelo_desnudo = suelo_desnudo.drop(["Muestra", "CE"], axis=1)

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
#polarimetria = polarimetria[['ENT', 'ALP', 'ANY', 'BMI', 'CSI']]

dataset = pd.concat([opticos, 
                     polarimetria
                     ], 
                    axis=1)

#%% INDICES ÓPTICOS Y SAR
opticos = r"C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\indices_opticos_5_abr.csv"
opticos = pd.read_csv(opticos)
print("Columnas con valores NaN:", opticos.columns[opticos.isna().any()].tolist())
opticos = opticos[['SI1', 
                   'BI',
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

#%% MAPEO
ce_mapping = {
    (float('-inf'), 20): 1,
    (20, 100): 2,
    #(4, 6): 3,
    #(6, 8): 4,
    #(8, 16): 5,
    (100, float('inf')): 3
}

# Utilizar la función map para aplicar el mapeo a la columna 'CE' y crear una nueva columna 'CE_map'
df['CE_map'] = df['CE'].map(lambda x: next((v for k, v in ce_mapping.items() if k[0] < x <= k[1]), None))

ce_map_counts = df ['CE_map'].value_counts()
ce_map_counts.plot(kind='bar')
plt.xlabel('CE_map')
plt.ylabel('Frecuencia')
plt.show()

df = df.dropna()

#%%
"""
#%%
X = df.drop(columns=["CE", "CE_map"])
y = df["CE_map"]

#%% 
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=123)

#%%
modelo = RandomForestClassifier(
                oob_score    = True,
                n_jobs       = -1,
                random_state = 123,
             )
    
#%%
modelo.fit(X_train, y_train)
"""

#%%
# División de los datos en train y test
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(
                                        df.drop(columns = ['CE_map', 'CE']),
                                        df['CE_map'],
                                        random_state = 123
                                    )

# One-hot-encoding de las variables categóricas
# ==============================================================================
# Se identifica el nobre de las columnas numéricas y categóricas
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.to_list()
numeric_cols = X_train.select_dtypes(include=['float64', 'int']).columns.to_list()

# Se aplica one-hot-encoding solo a las columnas categóricas
preprocessor = ColumnTransformer(
                    [('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'), cat_cols)],
                    remainder='passthrough',
                    verbose_feature_names_out=False
               ).set_output(transform="pandas")

# Una vez que se ha definido el objeto ColumnTransformer, con el método fit()
# se aprenden las transformaciones con los datos de entrenamiento y se aplican a
# los dos conjuntos con transform(). Ambas operaciones a la vez con fit_transform().
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep  = preprocessor.transform(X_test)

#%%
X_train_prep.info()

#%% Grid Search basado en validación cruzada
# Grid de hiperparámetros evaluados
# ==============================================================================
param_grid = {
    'n_estimators': [150],
    'max_features': [5, 7, 9],
    'max_depth'   : [None, 3, 10, 20],
    'criterion'   : ['gini', 'entropy']
}

# Búsqueda por grid search con validación cruzada
# ==============================================================================
grid = GridSearchCV(
        estimator  = RandomForestClassifier(random_state = 123),
        param_grid = param_grid,
        scoring    = 'accuracy',
        n_jobs     = multiprocessing.cpu_count() - 1,
        cv         = RepeatedKFold(n_splits=5, n_repeats=3, random_state=123), 
        refit      = True,
        verbose    = 0,
        return_train_score = True
       )

grid.fit(X = X_train_prep, y = y_train)

# Resultados
# ==============================================================================
resultados = pd.DataFrame(grid.cv_results_)
resultados.filter(regex = '(param*|mean_t|std_t)') \
    .drop(columns = 'params') \
    .sort_values('mean_test_score', ascending = False) \
    .head(4)

#%%
# Mejores hiperparámetros por validación cruzada
# ==============================================================================
print("----------------------------------------")
print("Mejores hiperparámetros encontrados (cv)")
print("----------------------------------------")
print(grid.best_params_, ":", grid.best_score_, grid.scoring)

#%%
modelo_final = grid.best_estimator_

#%%
# Error de test del modelo final
# ==============================================================================
predicciones = modelo_final.predict(X = X_test_prep)
predicciones[:10]

#%%
mat_confusion = confusion_matrix(
                    y_true    = y_test,
                    y_pred    = predicciones
                )

accuracy = accuracy_score(
            y_true    = y_test,
            y_pred    = predicciones,
            normalize = True
           )

print("Matriz de confusión")
print("-------------------")
print(mat_confusion)
print("")
print(f"El accuracy de test es: {100 * accuracy} %")
fig, ax = plt.subplots(figsize=(3, 3))
ConfusionMatrixDisplay(mat_confusion).plot(ax=ax);

#%%
print(
    classification_report(
        y_true = y_test,
        y_pred = predicciones
    )
)

#%%
importancia_predictores = pd.DataFrame(
                            {'predictor': X_train_prep.columns,
                             'importancia': modelo_final.feature_importances_}
                            )
print("Importancia de los predictores en el modelo")
print("-------------------------------------------")
importancia_predictores.sort_values('importancia', ascending=False)

#%%
importancia = permutation_importance(
                estimator    = modelo_final,
                X            = X_train_prep,
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
df_importancia['feature'] = X_train_prep.columns
df_importancia.sort_values('importances_mean', ascending=False)

# Gráfico
fig, ax = plt.subplots(figsize=(10, 12))
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



