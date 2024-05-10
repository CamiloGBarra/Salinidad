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
from sklearn.metrics import accuracy_score

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
#CE = np.sqrt(CE)

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

#%% MAPEO
ce_mapping = {
    (float('-inf'), 2): 1,
    (2, 4): 2,
    (4, 6): 3,
    (6, 8): 4,
    (8, 16): 5,
    (16, float('inf')): 6
}

# Utilizar la función map para aplicar el mapeo a la columna 'CE' y crear una nueva columna 'CE_map'
df['CE_map'] = df['CE'].map(lambda x: next((v for k, v in ce_mapping.items() if k[0] < x <= k[1]), None))

ce_map_counts = df ['CE_map'].value_counts()
ce_map_counts.plot(kind='bar')
plt.xlabel('CE_map')
plt.ylabel('Frecuencia')
plt.show()

df = df.dropna()

df = df.drop(["CE"], axis=1)

#%%
"""
SVM LINEAL
"""
#%%
# División de los datos en train y test
# ==============================================================================
X = df.drop("CE_map", axis="columns")
y = df['CE_map']

X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
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

#%%
"""
SVM LINEAL
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

#%%
# Matriz de confusión de las predicciones de test
# ==============================================================================
confusion_matrix = pd.crosstab(
    y_test.ravel(),
    predicciones,
    rownames=['Real'],
    colnames=['Predicción']
)
confusion_matrix




















