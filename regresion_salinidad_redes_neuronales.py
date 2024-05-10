# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('fivethirtyeight')

# Modelado
# ==============================================================================
from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from sklearn import set_config
import multiprocessing

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

#%%
csv_muestreo_abril2023 = r"C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\suelo_abr2023.csv"
csv_muestreo_abril2023 = pd.read_csv(csv_muestreo_abril2023, sep=';', decimal='.')
#csv_muestreo_abril2023 = csv_muestreo_abril2023[['Muestra','CE', 'Humedad', 'RAS', 'pH', 'Suelo_desnudo_porc']]
CE = csv_muestreo_abril2023[['CE']]

#%%
opticos = r"C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\indices_opticos_5_abr.csv"
opticos = pd.read_csv(opticos)
print("Columnas con valores NaN:", opticos.columns[opticos.isna().any()].tolist())
#opticos = opticos[['SI1', 'NDVI']]

polarimetria = r"C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\indices_polarimetricos.csv"
polarimetria = pd.read_csv(polarimetria)
print("Columnas con valores NaN:", polarimetria.columns[polarimetria.isna().any()].tolist())
#polarimetria = polarimetria.iloc[:, 0:18]
#polarimetria = polarimetria[['PAU1', 'PAU1','IRV', 'ENT', 'ANY']]

dataset = pd.concat([polarimetria], axis=1)
#dataset = opticos
#dataset = polarimetria

#%%
df = pd.concat([CE, dataset], axis=1)

#%%
df.info()
df.isna().sum().sort_values()

# Distribución variable respuesta
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 3))
sns.histplot(data=df, x="CE", kde=True, ax=ax)
ax.set_title("Distribución CE")
ax.set_label("CE");

#%% 
X_train, X_test, y_train, y_test = train_test_split(
                                                    df.drop("CE", axis="columns"),
                                                    df['CE'],
                                                    train_size   = 0.8,
                                                    random_state = 123,
                                                    shuffle      = True
                                                    )
print("Partición de entrenamento")
print("-----------------------")
display(y_train.describe())
print(" ")

print("Partición de test")
print("-----------------------")
display(y_test.describe())

#%%
#Preprocesado
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%%
#Modelado
regressor = MLPRegressor(
                        hidden_layer_sizes   =  (10, 10), 
                        max_iter             =  1000, 
                        random_state         =  123
                        )

# Entrena el modelo
regressor.fit(X_train, y_train)

# Realiza predicciones en el conjunto de prueba
y_pred = regressor.predict(X_test)

# Evalúa el rendimiento del modelo

mse = mean_squared_error(
                            y_true  = y_test,
                            y_pred  = y_pred,
                            squared = False
                        )
print(f"El error cuadrático medio es: {mse}") 











