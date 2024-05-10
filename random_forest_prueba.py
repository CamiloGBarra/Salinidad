#%%
# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd
from tabulate import tabulate

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.ticker as ticker
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats

# Preprocesado y modelado
# ==============================================================================
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge

from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence
import optuna
import geopandas as gpd

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

from correlaciones_indices import df_indices

#%%
shapefile_path = "C://Users//camil//Downloads//python_salinidad//shapes//puntos.shp"
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
df = df_indices
columnas_deseadas = ["CE.2", "NDSI", "NDWI", "SI1", "NDVI"]
df = df_indices[columnas_deseadas]
#df = df_indices.drop(["SAI1", "SAI3", "SAI4", "SAI5", "BI", "TBI", "EVI", "SI3"], axis=1)
df = df.join(proximidad_acequias_columna)

df = df.rename(columns={'CE.2': 'CE'})

#%%
"""
ANÁLISIS EXPLORATORIO
"""
#%%
df.head(5)

#%%
#Número de datos ausentes por variable
df.isna().sum().sort_values

#%%
"""
Variable respuesta
Cuando se crea un modelo, es muy importante estudiar la distribución de la variable respuesta, 
ya que, a fin de cuentas, es lo que interesa predecir. 
"""
#%%
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(6, 8))

# Distribución original
sns.kdeplot(
    df['CE'],
    fill=True,
    color="blue",
    ax=axes[0]
)
sns.rugplot(
    df['CE'],
    color="blue",
    ax=axes[0]
)
axes[0].set_title("Distribución original", fontsize='medium')
axes[0].set_xlabel('CE', fontsize='small')
axes[0].tick_params(labelsize=6)

# Transformación raíz cuadrada
sns.kdeplot(
    np.sqrt(df['CE']),
    fill=True,
    color="blue",
    ax=axes[1]
)
sns.rugplot(
    np.sqrt(df['CE']),
    color="blue",
    ax=axes[1]
)
axes[1].set_title("Transformación raíz cuadrada", fontsize='medium')
axes[1].set_xlabel('raíz cuadrada(CE)', fontsize='small')
axes[1].tick_params(labelsize=6)

# Transformación logarítmica
sns.kdeplot(
    np.log(df['CE']),
    fill=True,
    color="blue",
    ax=axes[2]
)
sns.rugplot(
    np.log(df['CE']),
    color="blue",
    ax=axes[2]
)
axes[2].set_title("Transformación logarítmica", fontsize='medium')
axes[2].set_xlabel('log(CE)', fontsize='small')
axes[2].tick_params(labelsize=6)

# Distribución gamma
shape, loc, scale = stats.gamma.fit(df['CE'])
x = np.linspace(0, df['CE'].max(), 100)
pdf = stats.gamma.pdf(x, shape, loc, scale)
axes[3].plot(x, pdf, label='Distribución Gamma', color='red')
axes[3].set_title("Distribución Gamma", fontsize='medium')
axes[3].set_xlabel('CE', fontsize='small')
axes[3].tick_params(labelsize=6)
axes[3].legend()

fig.tight_layout()
plt.show()


#%%
# Existen varias librerías en python que permiten identificar a qué distribución 
# se ajustan mejor los datos, una de ellas es fitter. Esta librería permite ajustar 
# cualquiera de las 80 distribuciones implementadas en scipy.

distribuciones = ['cauchy', 'chi2', 'expon',  'exponpow', 'gamma',
                  'norm', 'powerlaw', 'beta', 'logistic']

fitter = Fitter(df.CE, distributions=distribuciones)
fitter.fit()
fitter.summary(Nbest=10, plot=True)

#%%
"""
VARIABLES NUMÉRICAS
"""
#%%
#Variables numéricas
estadistica = df.select_dtypes(include=['float64', 'int']).describe()

#Gráfico de distribución para cada variable numérica
#Ajustar número de subplots en función del número de columnas
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 5))
axes = axes.flat
columnas_numeric = df.select_dtypes(include=['float64', 'int']).columns
columnas_numeric = columnas_numeric.drop('CE')

for i, colum in enumerate(columnas_numeric):
    sns.histplot(
        data     = df,
        x        = colum,
        stat     = "count",
        kde      = True,
        color    = (list(plt.rcParams['axes.prop_cycle'])*2)[i]["color"],
        line_kws = {'linewidth': 2},
        alpha    = 0.3,
        ax       = axes[i]
    )
    axes[i].set_title(colum, fontsize = 7, fontweight = "bold")
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel("")
    
    
fig.tight_layout()
plt.subplots_adjust(top = 0.9)
fig.suptitle('Distribución variables numéricas', fontsize = 10, fontweight = "bold");

#%%
# Gráfico de distribución para cada variable numérica
# ==============================================================================
# Ajustar número de subplots en función del número de columnas
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 5))
axes = axes.flat
columnas_numeric = df.select_dtypes(include=['float64', 'int']).columns
columnas_numeric = columnas_numeric.drop('CE')

for i, colum in enumerate(columnas_numeric):
    sns.regplot(
        x           = df[colum],
        y           = df['CE'],
        color       = "gray",
        marker      = '.',
        scatter_kws = {"alpha":0.4},
        line_kws    = {"color":"r","alpha":0.7},
        ax          = axes[i]
    )
    axes[i].set_title(f"precio vs {colum}", fontsize = 7, fontweight = "bold")
    #axes[i].ticklabel_format(style='sci', scilimits=(-4,4), axis='both')
    axes[i].yaxis.set_major_formatter(ticker.EngFormatter())
    axes[i].xaxis.set_major_formatter(ticker.EngFormatter())
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")

# Se eliminan los axes vacíos
for i in [8]:
    fig.delaxes(axes[i])
    
fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Correlación con CE', fontsize = 10, fontweight = "bold");

#%%
"""
CORRELACIÓN CON VARIABLES NUMÉRICAS
"""
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
    annot_kws = {"size": 6},
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
DIVISIÓN TRAIN Y TEST
"""
#%%
# Reparto de datos en train y test
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(
                                        df.drop('CE', axis = 'columns'),
                                        df['CE'],
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )

#%%
print("Partición de entrenamento")
print("-----------------------")
print(y_train.describe())

#%%
print("Partición de test")
print("-----------------------")
print(y_test.describe())

#%%
"""
CREAR EL MODELO
"""
#%%
#ENTRENAMIENTO


































