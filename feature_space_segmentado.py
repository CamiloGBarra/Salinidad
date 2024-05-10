#%%
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import pearsonr, linregress
import itertools
import numpy as np

#%%
datos_campo = pd.read_csv(r"C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\planilla_completa.csv", sep=";", decimal=",")
ce = datos_campo[['CE', 'SD_porc']]

ce_suelo_desnudo = datos_campo[datos_campo['SD_porc'] > 50]
ce_alta_salinidad = datos_campo[datos_campo['CE']>45]
ce_baja_salinidad = datos_campo[datos_campo['CE']<180]

#%%
# 5 abril 2023
indices_opticos = pd.read_csv(r"C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\indices_opticos_5_abr.csv")
# Anual 2022-2023
#indices_opticos = pd.read_csv(r"C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\indices_opticos_anual_2022_2023.csv")

indices_polarimetricos_buffer_medias_abr2023 = pd.read_csv(r"C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\indices_polarimetricos_buffer_medias_abr2023.csv")
indices_polarimetricos_buffer_medias_abr2023.drop('FID', axis=1, inplace=True)

#%%
df = pd.concat([ce,
                indices_polarimetricos_buffer_medias_abr2023, 
                indices_opticos
                ], 
               axis=1)
columns_with_nan = df.columns[df.isnull().any()]
print("Columnas con valores NaN:")
for column in columns_with_nan:
    print(column)

#df = df.drop(columns=columns_with_nan)

suelo_desnudo = df[df['SD_porc'] > 50].drop(['CE', 'SD_porc'], axis=1)
alta_salinidad = df[df['CE']>45].drop(['CE', 'SD_porc'], axis=1)
baja_salinidad = df[df['CE']<180].drop(['CE', 'SD_porc'], axis=1)

#%%
"""
ALTA SALINIDAD (CE>45)
"""
#%% NORMALIZAR entre 0 y 1
# Configurar el estilo de fuente
font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}
plt.rc('font', **font)

# Crear el gráfico original
columnas = alta_salinidad.columns
fig, ax = plt.subplots()
alta_salinidad[columnas].boxplot(ax=ax)
ax.set_xticklabels(columnas, rotation=90, ha='right')
plt.xlabel('Índices', fontweight='bold')
plt.ylabel('Valores', fontweight='bold')
plt.title('Datos Originales', fontweight='bold', fontsize=16)
plt.show()

# Normalizar los datos
scaler = MinMaxScaler()
columnas = alta_salinidad.columns
alta_salinidad[columnas] = scaler.fit_transform(alta_salinidad[columnas])

# Crear el gráfico de los datos normalizados
columnas = alta_salinidad.columns
fig, ax = plt.subplots()
alta_salinidad[columnas].boxplot(ax=ax)
ax.set_xticklabels(columnas, rotation=90, ha='right')
plt.xlabel('Índices', fontweight='bold')
plt.ylabel('Valores', fontweight='bold') 
plt.title('Datos Normalizados de 0 a 1', fontweight='bold', fontsize=16)
plt.show()

#%%
columnas_df = alta_salinidad.columns
combinaciones = list(itertools.product(columnas_df, repeat=2))

nuevas_filas1 = []
nuevas_filas2 = []
nuevas_filas3 = []
nuevas_filas4 = []
nuevas_filas5 = []
for col1, col2 in combinaciones:
    resultado1 = (((alta_salinidad[col1] - 1)**2) + (alta_salinidad[col2]**2)) ** 0.5
    resultado2 = ((alta_salinidad[col1]**2) + alta_salinidad[col2]) ** 0.5
    resultado3 = (((1 - alta_salinidad[col1])**2) + alta_salinidad[col2]) ** 0.5
    resultado4 = ((alta_salinidad[col1]**2) + (alta_salinidad[col2]**2)) ** 0.5
    resultado5 = (((1 - alta_salinidad[col1])**2) + ((1 - alta_salinidad[col2])**2)) ** 0.5
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

                                                                                           #5 abril  #Anual 2022-2023 
transformacion = ce_alta_salinidad['CE']                                                  # 0,27    # 0,27
#transformacion = ce_alta_salinidad['log_CE'] = np.log(ce_alta_salinidad['CE'])             # 0,38    # 0,38
#transformacion = ce_alta_salinidad['sqrt_CE'] = ce_alta_salinidad['CE']**2                 # 0,42    # 0,42
#transformacion = ce_alta_salinidad['raiz_cuadrada_CE'] = np.sqrt(ce_alta_salinidad['CE'])  # 0,39    # 0,39
#transformacion = ce_alta_salinidad['inverso_CE'] = 1 / ce_alta_salinidad['CE']             # 0,45    # 0,45
#transformacion = ce_alta_salinidad['exp_CE'] = np.exp(ce_alta_salinidad['CE'])             # 0,29    # 0,41
#transformacion = ce_alta_salinidad['abs_CE'] = np.abs(ce_alta_salinidad['CE'])             # 0,40    # 0,41
#transformacion = ce_alta_salinidad['round_CE'] = np.round(ce_alta_salinidad['CE'])         # 0,40    # 0,40
#transformacion = ce_alta_salinidad['CE_elevado_a_3'] = ce_alta_salinidad['CE'] ** 3        # 0,42    # 0,39



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

    
    print(f"(((alta_salinidad[col1] - 1)**2) + (alta_salinidad[col2]**2)) ** 0.5 / 10 valores más altos para '{col_objetivo}':")
    print(correlaciones_ordenadas1.head(10))
    
    print(f"((alta_salinidad[col1]**2) + alta_salinidad[col2]) ** 0.5 / 10 valores más altos para '{col_objetivo}':")
    print(correlaciones_ordenadas2.head(10))
    
    print(f"(((1 - alta_salinidad[col1])**2) + alta_salinidad[col2]) ** 0.5 / 10 valores más altos para '{col_objetivo}':")
    print(correlaciones_ordenadas3.head(10))
    
    print(f"((alta_salinidad[col1]**2) + (alta_salinidad[col2]**2)) ** 0.5 / 10 valores más altos para '{col_objetivo}':")
    print(correlaciones_ordenadas4.head(10))
    
    print(f"(((1 - alta_salinidad[col1])**2) + ((1 - alta_salinidad[col2])**2)) ** 0.5 / 10 valores más altos para '{col_objetivo}':")
    print(correlaciones_ordenadas5.head(10))

#%%
"""
SUELO DESNUDO (SD mayor al 50%)
"""
#%% NORMALIZAR entre 0 y 1
# Configurar el estilo de fuente
font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}
plt.rc('font', **font)

# Crear el gráfico original
columnas = suelo_desnudo.columns
fig, ax = plt.subplots()
suelo_desnudo[columnas].boxplot(ax=ax)
ax.set_xticklabels(columnas, rotation=90, ha='right')
plt.xlabel('Índices', fontweight='bold')
plt.ylabel('Valores', fontweight='bold')
plt.title('Datos Originales', fontweight='bold', fontsize=16)
plt.show()

# Normalizar los datos
scaler = MinMaxScaler()
columnas = suelo_desnudo.columns
suelo_desnudo[columnas] = scaler.fit_transform(suelo_desnudo[columnas])

# Crear el gráfico de los datos normalizados
columnas = suelo_desnudo.columns
fig, ax = plt.subplots()
suelo_desnudo[columnas].boxplot(ax=ax)
ax.set_xticklabels(columnas, rotation=90, ha='right')
plt.xlabel('Índices', fontweight='bold')
plt.ylabel('Valores', fontweight='bold') 
plt.title('Datos Normalizados de 0 a 1', fontweight='bold', fontsize=16)
plt.show()

#%%
columnas_df = suelo_desnudo.columns
combinaciones = list(itertools.product(columnas_df, repeat=2))

nuevas_filas1 = []
nuevas_filas2 = []
nuevas_filas3 = []
nuevas_filas4 = []
nuevas_filas5 = []
for col1, col2 in combinaciones:
    resultado1 = (((suelo_desnudo[col1] - 1)**2) + (suelo_desnudo[col2]**2)) ** 0.5
    resultado2 = ((suelo_desnudo[col1]**2) + suelo_desnudo[col2]) ** 0.5
    resultado3 = (((1 - suelo_desnudo[col1])**2) + suelo_desnudo[col2]) ** 0.5
    resultado4 = ((suelo_desnudo[col1]**2) + (suelo_desnudo[col2]**2)) ** 0.5
    resultado5 = (((1 - suelo_desnudo[col1])**2) + ((1 - suelo_desnudo[col2])**2)) ** 0.5
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

                                                                                          #5 abril  #Anual 2022-2023 
transformacion = ce_suelo_desnudo['CE']                                                  # 0,48    # 0,49
#transformacion = ce_suelo_desnudo['log_CE'] = np.log(ce_suelo_desnudo['CE'])             # 0,43    # 0,47
#transformacion = ce_suelo_desnudo['sqrt_CE'] = ce_suelo_desnudo['CE']**2                 # 0,47    # 0,47
#transformacion = ce_suelo_desnudo['raiz_cuadrada_CE'] = np.sqrt(ce_suelo_desnudo['CE'])  # 0,48    # 0,48
#transformacion = ce_suelo_desnudo['inverso_CE'] = 1 / ce_suelo_desnudo['CE']             # 0,44    # 0,44
#transformacion = ce_suelo_desnudo['exp_CE'] = np.exp(ce_suelo_desnudo['CE'])             # 0,24    # 0,43
#transformacion = ce_suelo_desnudo['abs_CE'] = np.abs(ce_suelo_desnudo['CE'])             # 0,49    # 0,49
#transformacion = ce_suelo_desnudo['round_CE'] = np.round(ce_suelo_desnudo['CE'])         # 0,49    # 0,49
#transformacion = ce_suelo_desnudo['CE_elevado_a_3'] = ce_suelo_desnudo['CE'] ** 3        # 0,44    # 0,44



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

    
    print(f"(((suelo_desnudo[col1] - 1)**2) + (suelo_desnudo[col2]**2)) ** 0.5 / 10 valores más altos para '{col_objetivo}':")
    print(correlaciones_ordenadas1.head(10))
    
    print(f"((suelo_desnudo[col1]**2) + suelo_desnudo[col2]) ** 0.5 / 10 valores más altos para '{col_objetivo}':")
    print(correlaciones_ordenadas2.head(10))
    
    print(f"(((1 - suelo_desnudo[col1])**2) + suelo_desnudo[col2]) ** 0.5 / 10 valores más altos para '{col_objetivo}':")
    print(correlaciones_ordenadas3.head(10))
    
    print(f"((suelo_desnudo[col1]**2) + (suelo_desnudo[col2]**2)) ** 0.5 / 10 valores más altos para '{col_objetivo}':")
    print(correlaciones_ordenadas4.head(10))
    
    print(f"(((1 - suelo_desnudo[col1])**2) + ((1 - suelo_desnudo[col2])**2)) ** 0.5 / 10 valores más altos para '{col_objetivo}':")
    print(correlaciones_ordenadas5.head(10))

#%%
"""
BAJA SALINIDAD (CE<45)
"""
#%% NORMALIZAR entre 0 y 1
# Configurar el estilo de fuente
font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}
plt.rc('font', **font)

# Crear el gráfico original
columnas = baja_salinidad.columns
fig, ax = plt.subplots()
baja_salinidad[columnas].boxplot(ax=ax)
ax.set_xticklabels(columnas, rotation=90, ha='right')
plt.xlabel('Índices', fontweight='bold')
plt.ylabel('Valores', fontweight='bold')
plt.title('Datos Originales', fontweight='bold', fontsize=16)
plt.show()

# Normalizar los datos
scaler = MinMaxScaler()
columnas = baja_salinidad.columns
baja_salinidad[columnas] = scaler.fit_transform(baja_salinidad[columnas])

# Crear el gráfico de los datos normalizados
columnas = baja_salinidad.columns
fig, ax = plt.subplots()
baja_salinidad[columnas].boxplot(ax=ax)
ax.set_xticklabels(columnas, rotation=90, ha='right')
plt.xlabel('Índices', fontweight='bold')
plt.ylabel('Valores', fontweight='bold') 
plt.title('Datos Normalizados de 0 a 1', fontweight='bold', fontsize=16)
plt.show()

#%%
columnas_df = baja_salinidad.columns
combinaciones = list(itertools.product(columnas_df, repeat=2))

nuevas_filas1 = []
nuevas_filas2 = []
nuevas_filas3 = []
nuevas_filas4 = []
nuevas_filas5 = []
for col1, col2 in combinaciones:
    resultado1 = (((baja_salinidad[col1] - 1)**2) + (baja_salinidad[col2]**2)) ** 0.5
    resultado2 = ((baja_salinidad[col1]**2) + baja_salinidad[col2]) ** 0.5
    resultado3 = (((1 - baja_salinidad[col1])**2) + baja_salinidad[col2]) ** 0.5
    resultado4 = ((baja_salinidad[col1]**2) + (baja_salinidad[col2]**2)) ** 0.5
    resultado5 = (((1 - baja_salinidad[col1])**2) + ((1 - baja_salinidad[col2])**2)) ** 0.5
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

                                                                                           #5 abril  #Anual 2022-2023 
#transformacion = ce_baja_salinidad['CE']                                                  # 0,55    # 0,59
transformacion = ce_baja_salinidad['log_CE'] = np.log(ce_baja_salinidad['CE'])             # 0,91    # 0,91
#transformacion = ce_baja_salinidad['sqrt_CE'] = ce_baja_salinidad['CE']**2                 # 0,93    # 0,93
#transformacion = ce_baja_salinidad['raiz_cuadrada_CE'] = np.sqrt(ce_baja_salinidad['CE'])  # 0,88    # 0,87
#transformacion = ce_baja_salinidad['inverso_CE'] = 1 / ce_baja_salinidad['CE']             # 0,88    # 0,85
#transformacion = ce_baja_salinidad['exp_CE'] = np.exp(ce_baja_salinidad['CE'])             # 0,94    # 0,94
#transformacion = ce_baja_salinidad['abs_CE'] = np.abs(ce_baja_salinidad['CE'])             # 0,91    # 0,91
#transformacion = ce_baja_salinidad['round_CE'] = np.round(ce_baja_salinidad['CE'])         # 0,90    # 0,90
#transformacion = ce_baja_salinidad['CE_elevado_a_3'] = ce_baja_salinidad['CE'] ** 3        # 0,93    # 0,93



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

    
    print(f"(((baja_salinidad[col1] - 1)**2) + (baja_salinidad[col2]**2)) ** 0.5 / 10 valores más altos para '{col_objetivo}':")
    print(correlaciones_ordenadas1.head(10))
    
    print(f"((baja_salinidad[col1]**2) + baja_salinidad[col2]) ** 0.5 / 10 valores más altos para '{col_objetivo}':")
    print(correlaciones_ordenadas2.head(10))
    
    print(f"(((1 - baja_salinidad[col1])**2) + baja_salinidad[col2]) ** 0.5 / 10 valores más altos para '{col_objetivo}':")
    print(correlaciones_ordenadas3.head(10))
    
    print(f"((baja_salinidad[col1]**2) + (baja_salinidad[col2]**2)) ** 0.5 / 10 valores más altos para '{col_objetivo}':")
    print(correlaciones_ordenadas4.head(10))
    
    print(f"(((1 - baja_salinidad[col1])**2) + ((1 - baja_salinidad[col2])**2)) ** 0.5 / 10 valores más altos para '{col_objetivo}':")
    print(correlaciones_ordenadas5.head(10))

#%% 
"""
la correlación elegida fue (((1 - GARI)**2) + ((1 - VSI)**2)) ** 0.5  
(GARI, VSI) ---> 0.764
"""
df = df[df['CE']<65]
gari_vsi = df[["GARI", "VSI"]]

columnas = gari_vsi.columns
fig, ax = plt.subplots()
gari_vsi[columnas].boxplot(ax=ax)
ax.set_xticklabels(columnas, rotation=90, ha='right')
plt.xlabel('Índices')
plt.ylabel('Valores')
plt.title('Datos Originales')
plt.show()

for columna in gari_vsi.columns:
    valor_minimo = gari_vsi[columna].min()
    valor_maximo = gari_vsi[columna].max()
    gari_vsi[columna] = (gari_vsi[columna] - valor_minimo) / (valor_maximo - valor_minimo)

columnas = gari_vsi.columns
fig, ax = plt.subplots()
gari_vsi[columnas].boxplot(ax=ax)
ax.set_xticklabels(columnas, rotation=90, ha='right')
plt.xlabel('Índices')
plt.ylabel('Valores')
plt.title('Datos Normalizados de 0 a 1')
plt.show()


fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(gari_vsi["GARI"], gari_vsi["VSI"], estilo, color=color)
plt.xlabel("GARI")
plt.ylabel("VSI")
plt.show()

log_CE = np.log(df['CE'])

gari_vsi_CE = pd.DataFrame({'GARI_VSI': (((1 - gari_vsi['GARI'])**2) + ((1 - gari_vsi['VSI'])**2)) ** 0.5,
                         'CE': log_CE}).dropna()

corr_pearson = gari_vsi_CE["GARI_VSI"].corr(gari_vsi_CE["CE"], method="pearson")
print(f"Correlación de Pearson GARI_VSI_CE: {corr_pearson}")

fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(gari_vsi_CE["CE"], gari_vsi_CE["GARI_VSI"], estilo, color=color)
plt.xlabel("log(CE)")
plt.ylabel("Índice Feature Space (GARI_VSI)")
plt.show()

#%%
CE = gari_vsi_CE['CE'] #esto en realidad es el log(CE)
GARI_VSI = gari_vsi_CE['GARI_VSI']

slope, intercept, r_value, p_value, std_err = linregress(CE, GARI_VSI)

rango_CE = np.linspace(min(CE), max(CE), 100)
GARI_VSI_fit = slope * rango_CE + intercept

fig, ax = plt.subplots()
estilo = "o"
color = "steelblue"
ax.plot(CE, GARI_VSI, estilo, color=color)
ax.plot(rango_CE, GARI_VSI_fit, color="red", label=f"Regresión lineal: y = {slope:.2f}x + {intercept:.2f}")
plt.xlabel("log(CE)")
plt.ylabel("Índice Feature Space (GARI_VSI)")
plt.legend()
plt.show()

formula = f"Regresión lineal: log(CE) = {slope:.2f}(GARI_VSI) + {intercept:.2f}"
print(formula)

#%%
import rasterio

EVI = r'C:\Users\camil\Downloads\Salinidad\imagenes SALINIDAD\índices ópticos S2\EVI.tif'
MNDWI = r'C:\Users\camil\Downloads\Salinidad\imagenes SALINIDAD\índices ópticos S2\MNDWI.tif'

with rasterio.open(EVI) as src1, rasterio.open(MNDWI) as src2:
    data1 = src1.read()
    data2 = src2.read()
    normalizado_evi = (data1.astype(float) - np.min(data1)) / (np.max(data1) - np.min(data1))
    normalizado_mndwi = (data2.astype(float) - np.min(data2)) / (np.max(data2) - np.min(data2))


resultado = (((1 - normalizado_evi) ** 2) + normalizado_mndwi) ** 0.5

log_ce = 0.09 * resultado + 0.65

ce = np.exp(log_ce)

output_path = r'C:\Users\camil\Downloads\CE_final.tif'

with rasterio.open(EVI) as src1:
    datos = src1.profile

with rasterio.open(output_path, 'w', **datos) as dst:
    dst.write(ce.astype(datos['dtype']))













