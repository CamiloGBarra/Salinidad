#%%
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import pearsonr, linregress
import itertools
import numpy as np

#%%
datos_campo = pd.read_csv(r"C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\planilla_completa.csv", sep=";", decimal=",")

variable = 'CE'
valor_minimo = 0
valor_maximo = 30

ce_rango = datos_campo[(datos_campo[variable] >= valor_minimo) & (datos_campo[variable] <= valor_maximo)]

#%%
indices_opticos = pd.read_csv(r"C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\indices_opticos_5_abr.csv")

indices_polarimetricos_buffer_medias_abr2023 = pd.read_csv(r"C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\indices_polarimetricos_buffer_medias_abr2023.csv")
indices_polarimetricos_buffer_medias_abr2023.drop('FID', axis=1, inplace=True)

#%%
ce = datos_campo[['CE', 'SD_porc']]
df = pd.concat([ce,
                indices_polarimetricos_buffer_medias_abr2023, 
                indices_opticos
                ], 
               axis=1)
columns_with_nan = df.columns[df.isnull().any()]
print("Columnas con valores NaN:")
for column in columns_with_nan:
    print(column)

rango = df[(df[variable] >= valor_minimo) & (df[variable] <= valor_maximo)].drop(['CE', 'SD_porc'], axis=1)

#%%
"""
Feature Space
"""
#%% NORMALIZAR entre 0 y 1
# Configurar el estilo de fuente
font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}
plt.rc('font', **font)

# Crear el gráfico original
columnas = rango.columns
fig, ax = plt.subplots()
rango[columnas].boxplot(ax=ax)
ax.set_xticklabels(columnas, rotation=90, ha='right')
plt.xlabel('Índices', fontweight='bold')
plt.ylabel('Valores', fontweight='bold')
plt.title('Datos Originales', fontweight='bold', fontsize=16)
plt.show()

# Normalizar los datos
scaler = MinMaxScaler()
columnas = rango.columns
rango[columnas] = scaler.fit_transform(rango[columnas])

# Crear el gráfico de los datos normalizados
columnas = rango.columns
fig, ax = plt.subplots()
rango[columnas].boxplot(ax=ax)
ax.set_xticklabels(columnas, rotation=90, ha='right')
plt.xlabel('Índices', fontweight='bold')
plt.ylabel('Valores', fontweight='bold') 
plt.title('Datos Normalizados de 0 a 1', fontweight='bold', fontsize=16)
plt.show()

#%%
columnas_df = rango.columns
combinaciones = list(itertools.product(columnas_df, repeat=2))

nuevas_filas1 = []
nuevas_filas2 = []
nuevas_filas3 = []
nuevas_filas4 = []
nuevas_filas5 = []
for col1, col2 in combinaciones:
    resultado1 = (((rango[col1] - 1)**2) + (rango[col2]**2)) ** 0.5
    resultado2 = ((rango[col1]**2) + rango[col2]) ** 0.5
    resultado3 = (((1 - rango[col1])**2) + rango[col2]) ** 0.5
    resultado4 = ((rango[col1]**2) + (rango[col2]**2)) ** 0.5
    resultado5 = (((1 - rango[col1])**2) + ((1 - rango[col2])**2)) ** 0.5
    nuevas_filas1.append(resultado1)
    nuevas_filas2.append(resultado2)
    nuevas_filas3.append(resultado3)
    nuevas_filas4.append(resultado4)
    nuevas_filas5.append(resultado5)

columnas_objetivo = ['CE']

nuevo_dataframe1 = pd.DataFrame(nuevas_filas1).transpose()
nuevo_dataframe2 = pd.DataFrame(nuevas_filas2).transpose()
nuevo_dataframe3 = pd.DataFrame(nuevas_filas3).transpose()
nuevo_dataframe4 = pd.DataFrame(nuevas_filas4).transpose()
nuevo_dataframe5 = pd.DataFrame(nuevas_filas5).transpose()

nuevo_dataframe1.columns = [f'({col1}, {col2})' for col1, col2 in combinaciones]
nuevo_dataframe2.columns = [f'({col1}, {col2})' for col1, col2 in combinaciones]
nuevo_dataframe3.columns = [f'({col1}, {col2})' for col1, col2 in combinaciones]
nuevo_dataframe4.columns = [f'({col1}, {col2})' for col1, col2 in combinaciones]
nuevo_dataframe5.columns = [f'({col1}, {col2})' for col1, col2 in combinaciones]


transformacion = ce_rango['log_CE'] = np.log(ce_rango['CE'])             


for col_objetivo in columnas_objetivo:
    correlaciones1 = nuevo_dataframe1.corrwith(transformacion, method='pearson')
    correlaciones2 = nuevo_dataframe2.corrwith(transformacion, method='pearson')
    correlaciones3 = nuevo_dataframe3.corrwith(transformacion, method='pearson')
    correlaciones4 = nuevo_dataframe4.corrwith(transformacion, method='pearson')
    correlaciones5 = nuevo_dataframe5.corrwith(transformacion, method='pearson')

    correlaciones_ordenadas1 = correlaciones1.sort_values(ascending=False)
    correlaciones_ordenadas2 = correlaciones2.sort_values(ascending=False)
    correlaciones_ordenadas3 = correlaciones3.sort_values(ascending=False)
    correlaciones_ordenadas4 = correlaciones4.sort_values(ascending=False)
    correlaciones_ordenadas5 = correlaciones5.sort_values(ascending=False)

    
    print(f"(((rango[col1] - 1)**2) + (rango[col2]**2)) ** 0.5 / 10 valores más altos para '{col_objetivo}':")
    print(correlaciones_ordenadas1.head(10))
    
    print(f"((rango[col1]**2) + rango[col2]) ** 0.5 / 10 valores más altos para '{col_objetivo}':")
    print(correlaciones_ordenadas2.head(10))
    
    print(f"(((1 - rango[col1])**2) + rango[col2]) ** 0.5 / 10 valores más altos para '{col_objetivo}':")
    print(correlaciones_ordenadas3.head(10))
    
    print(f"((rango[col1]**2) + (rango[col2]**2)) ** 0.5 / 10 valores más altos para '{col_objetivo}':")
    print(correlaciones_ordenadas4.head(10))
    
    print(f"(((1 - rango[col1])**2) + ((1 - rango[col2])**2)) ** 0.5 / 10 valores más altos para '{col_objetivo}':")
    print(correlaciones_ordenadas5.head(10))
