#%%
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import pearsonr, linregress
import itertools
import numpy as np

#%%
datos_campo = pd.read_csv(r"C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\planilla_completa.csv", sep=";", decimal=",")

#%%
# 5 abril 2023
indices_opticos = pd.read_csv(r"C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\indices_opticos_5_abr.csv")
# Anual 2022-2023
#indices_opticos = pd.read_csv(r"C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\indices_opticos_anual_2022_2023.csv")

indices_polarimetricos_buffer_medias_abr2023 = pd.read_csv(r"C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\indices_polarimetricos_buffer_medias_abr2023.csv")
indices_polarimetricos_buffer_medias_abr2023.drop('FID', axis=1, inplace=True)

#%%
df = pd.concat([
                indices_polarimetricos_buffer_medias_abr2023, 
                indices_opticos
                ], 
               axis=1)
columns_with_nan = df.columns[df.isnull().any()]
print("Columnas con valores NaN:")
for column in columns_with_nan:
    print(column)

df = df.drop(columns=columns_with_nan)

#%%
"""
ÓPTICOS
"""

#%%  ANTES
columnas = indices_opticos.columns
fig, ax = plt.subplots()
indices_opticos[columnas].boxplot(ax=ax)
ax.set_xticklabels(columnas, rotation=90, ha='right')
plt.xlabel('Índices')
plt.ylabel('Valores')
plt.show()

#%% normalización
scaler = StandardScaler()
columnas = indices_opticos.columns
indices_opticos[columnas] = scaler.fit_transform(indices_opticos[columnas])

#%%  DESPUÉS
columnas = indices_opticos.columns
fig, ax = plt.subplots()
indices_opticos[columnas].boxplot(ax=ax)
ax.set_xticklabels(columnas, rotation=90, ha='right')
plt.xlabel('Índices')
plt.ylabel('Valores')
plt.show()

#%%
fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(indices_opticos["NDVI"], indices_opticos["BI"], estilo, color=color)
plt.xlabel("NDVI")
plt.ylabel("BI")
plt.show()

fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(indices_opticos["SI1"], indices_opticos["BI"], estilo, color=color)
plt.xlabel("SI1")
plt.ylabel("BI")
plt.show()

fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(indices_opticos["NDVI"], indices_opticos["SI1"], estilo, color=color)
plt.xlabel("NDVI")
plt.ylabel("SI1")
plt.show()

fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(indices_opticos["NDSI"], indices_opticos["VSSI"], estilo, color=color)
plt.xlabel("NDSI")
plt.ylabel("VSSI")
plt.show()

fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(indices_opticos["NDSI"], indices_opticos["CORSI"], estilo, color=color)
plt.xlabel("NDSI")
plt.ylabel("CORSI")
plt.show()

fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(indices_opticos["MNDWI"], indices_opticos["CRSI"], estilo, color=color)
plt.xlabel("MNDWI")
plt.ylabel("CRSI")
plt.show()

#%%
NDVI_BI = pd.DataFrame({'NDVI_BI': (((1-indices_opticos['BI'])**2) + (indices_opticos['NDVI'])) ** (1/2),
                         'CE': datos_campo['CE']}).dropna()

SI1_BI = pd.DataFrame({'SI1_BI': ((indices_opticos["SI1"]**2) + indices_opticos["BI"]) ** (1/2),
                         'CE': datos_campo['CE']}).dropna()

NDVI_SI1 = pd.DataFrame({'NDVI_SI1': (((indices_opticos["NDVI"]-1)**2) + (indices_opticos["SI1"]**2)) ** (1/2),
                         'CE': datos_campo['CE']}).dropna()

VSSI_NDSI = pd.DataFrame({'VSSI_NDSI': (((1-indices_opticos["NDSI"])**2) + (indices_opticos["VSSI"]**2)) ** (1/2),
                         'CE': datos_campo['CE']}).dropna()

NDSI_CORSI = pd.DataFrame({'NDSI_CORSI': (((indices_opticos["NDSI"])**2) + (indices_opticos["CORSI"])) ** (1/2),
                         'CE': datos_campo['CE']}).dropna()

CRSI_MNDWI = pd.DataFrame({'CRSI_MNDWI': (((1-indices_opticos["MNDWI"])**2) + (indices_opticos["CRSI"])) ** (1/2),
                         'CE': datos_campo['CE']}).dropna()

#%% CORRELACIONES - PEARSON Y SPEARMAN

######################  NDVI-BI ####################################
corr_pearson = NDVI_BI["NDVI_BI"].corr(NDVI_BI["CE"], method="pearson")
corr_spearman = NDVI_BI["NDVI_BI"].corr(NDVI_BI["CE"], method="spearman")
corr_pearson_4 = pearsonr(x = NDVI_BI["NDVI_BI"], y =  NDVI_BI["CE"])
print(f"Correlación de Pearson NDVI_BI-CE: {corr_pearson}")
print(f"Correlación de Spearman NDVI_BI-CE: {corr_spearman}")
print("P-value: ", corr_pearson_4[1])

fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(NDVI_BI["CE"], NDVI_BI["NDVI_BI"], estilo, color=color)
plt.xlabel("CE")
plt.ylabel("NDVI_BI")
plt.show()

######################  SI1-BI ####################################
corr_pearson = SI1_BI["SI1_BI"].corr(SI1_BI["CE"], method="pearson")
corr_spearman = SI1_BI["SI1_BI"].corr(SI1_BI["CE"], method="spearman")
corr_pearson_3 = pearsonr(x = SI1_BI["SI1_BI"], y =  SI1_BI["CE"])
print(f"Correlación de Pearson SI1_BI-CE: {corr_pearson}")
print(f"Correlación de Spearman SI1_BI-CE: {corr_spearman}")
print("P-value: ", corr_pearson_3[1])

fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(SI1_BI["CE"], SI1_BI["SI1_BI"], estilo, color=color)
plt.xlabel("CE")
plt.ylabel("SI1_BI")
plt.show()

######################  NDVI-SI1 ####################################
corr_pearson = NDVI_SI1["NDVI_SI1"].corr(NDVI_SI1["CE"], method="pearson")
corr_spearman = NDVI_SI1["NDVI_SI1"].corr(NDVI_SI1["CE"], method="spearman")
corr_pearson_2 = pearsonr(x = NDVI_SI1["NDVI_SI1"], y =  NDVI_SI1["CE"])
print(f"Correlación de Pearson NDVI_SI1-CE: {corr_pearson}")
print(f"Correlación de Spearman NDVI_SI1-CE: {corr_spearman}")
print("P-value: ", corr_pearson_2[1])

fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(NDVI_SI1["CE"], NDVI_SI1["NDVI_SI1"], estilo, color=color)
plt.xlabel("CE")
plt.ylabel("NDVI_SI1")
plt.show()

######################  VSSI-NDSI ####################################
corr_pearson = VSSI_NDSI["VSSI_NDSI"].corr(VSSI_NDSI["CE"], method="pearson")
corr_spearman = VSSI_NDSI["VSSI_NDSI"].corr(VSSI_NDSI["CE"], method="spearman")
corr_pearson_1 = pearsonr(x = VSSI_NDSI["VSSI_NDSI"], y =  VSSI_NDSI["CE"])
print("P-value: ", corr_pearson_1[1])

print(f"Correlación de Pearson VSSI_NDSI-CE: {corr_pearson}")
print(f"Correlación de Spearman VSSI_NDSI-CE: {corr_spearman}")

fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(VSSI_NDSI["CE"], VSSI_NDSI["VSSI_NDSI"], estilo, color=color)
plt.xlabel("CE")
plt.ylabel("VSSI_NDSI")
plt.show()

######################  NDSI-CORSI ####################################
corr_pearson = NDSI_CORSI["NDSI_CORSI"].corr(NDSI_CORSI["CE"], method="pearson")
corr_spearman = NDSI_CORSI["NDSI_CORSI"].corr(NDSI_CORSI["CE"], method="spearman")
corr_pearson_1 = pearsonr(x = NDSI_CORSI["NDSI_CORSI"], y =  NDSI_CORSI["CE"])
print("P-value: ", corr_pearson_1[1])

print(f"Correlación de Pearson NDSI_CORSI-CE: {corr_pearson}")
print(f"Correlación de Spearman NDSI_CORSI-CE: {corr_spearman}")

fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(NDSI_CORSI["CE"], NDSI_CORSI["NDSI_CORSI"], estilo, color=color)
plt.xlabel("CE")
plt.ylabel("NDSI_CORSI")
plt.show()

######################  CRSI-MNDWI ####################################
corr_pearson = CRSI_MNDWI["CRSI_MNDWI"].corr(CRSI_MNDWI["CE"], method="pearson")
corr_spearman = CRSI_MNDWI["CRSI_MNDWI"].corr(CRSI_MNDWI["CE"], method="spearman")
corr_pearson_1 = pearsonr(x = CRSI_MNDWI["CRSI_MNDWI"], y =  CRSI_MNDWI["CE"])
print("P-value: ", corr_pearson_1[1])

print(f"Correlación de Pearson CRSI_MNDWI-CE: {corr_pearson}")
print(f"Correlación de Spearman CRSI_MNDWI-CE: {corr_spearman}")

fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(CRSI_MNDWI["CE"], CRSI_MNDWI["CRSI_MNDWI"], estilo, color=color)
plt.xlabel("CE")
plt.ylabel("CRSI_MNDWI")
plt.show()

#%%
"""
SAOCOM
"""

#%%  ANTES
columnas = descomposiciones.columns
fig, ax = plt.subplots()
descomposiciones[columnas].boxplot(ax=ax)
ax.set_xticklabels(columnas, rotation=90, ha='right')
plt.xlabel('Índices')
plt.ylabel('Valores')
plt.show()

#%% normalización
scaler = StandardScaler()
columnas = descomposiciones.columns
descomposiciones[columnas] = scaler.fit_transform(descomposiciones[columnas])

#%%  DESPUÉS
columnas = descomposiciones.columns
fig, ax = plt.subplots()
descomposiciones[columnas].boxplot(ax=ax)
ax.set_xticklabels(columnas, rotation=90, ha='right')
plt.xlabel('Índices')
plt.ylabel('Valores')
plt.show()

#%%
fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(descomposiciones["Sinclair_2"], descomposiciones["FreemanDurden_2"], estilo, color=color)
plt.xlabel("Sinclair_2")
plt.ylabel("FreemanDurden_2")
plt.show()

fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(descomposiciones["vanZyl_2"], descomposiciones["Yang_2"], estilo, color=color)
plt.xlabel("vanZyl_2")
plt.ylabel("Yang_2")
plt.show()

#%%
Sinclair_2_FreemanDurden_2 = pd.DataFrame({'Sinclair_2_FreemanDurden_2': (((1-descomposiciones['Sinclair_2'])**2) + ((1-descomposiciones['FreemanDurden_2'])**2)) ** (1/2),
                         'CE': datos_campo['CE']}).dropna()

vanZyl_2_Yang_2 = pd.DataFrame({'vanZyl_2_Yang_2': (((1-descomposiciones['vanZyl_2'])**2) + ((1-descomposiciones['Yang_2'])**2)) ** (1/2),
                         'CE': datos_campo['CE']}).dropna()

#%% CORRELACIONES - PEARSON Y SPEARMAN

######################  Sinclair_2-FreemanDurden_2 ####################################
corr_pearson = Sinclair_2_FreemanDurden_2["Sinclair_2_FreemanDurden_2"].corr(Sinclair_2_FreemanDurden_2["CE"], method="pearson")
corr_spearman = Sinclair_2_FreemanDurden_2["Sinclair_2_FreemanDurden_2"].corr(Sinclair_2_FreemanDurden_2["CE"], method="spearman")
corr_pearson_4 = pearsonr(x = Sinclair_2_FreemanDurden_2["Sinclair_2_FreemanDurden_2"], y =  Sinclair_2_FreemanDurden_2["CE"])
print(f"Correlación de Pearson Sinclair_2_FreemanDurden_2: {corr_pearson}")
print(f"Correlación de Spearman Sinclair_2_FreemanDurden_2: {corr_spearman}")
print("P-value: ", corr_pearson_4[1])

fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(Sinclair_2_FreemanDurden_2["CE"], Sinclair_2_FreemanDurden_2["Sinclair_2_FreemanDurden_2"], estilo, color=color)
plt.xlabel("CE")
plt.ylabel("Sinclair_2_FreemanDurden_2")
plt.show()

######################  vanZyl_2-Yang_2 ####################################
corr_pearson = vanZyl_2_Yang_2["vanZyl_2_Yang_2"].corr(vanZyl_2_Yang_2["CE"], method="pearson")
corr_spearman = vanZyl_2_Yang_2["vanZyl_2_Yang_2"].corr(vanZyl_2_Yang_2["CE"], method="spearman")
corr_pearson_4 = pearsonr(x = vanZyl_2_Yang_2["vanZyl_2_Yang_2"], y =  vanZyl_2_Yang_2["CE"])
print(f"Correlación de Pearson vanZyl_2_Yang_2: {corr_pearson}")
print(f"Correlación de Spearman vanZyl_2_Yang_2: {corr_spearman}")
print("P-value: ", corr_pearson_4[1])

fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(vanZyl_2_Yang_2["CE"], vanZyl_2_Yang_2["vanZyl_2_Yang_2"], estilo, color=color)
plt.xlabel("CE")
plt.ylabel("vanZyl_2_Yang_2")
plt.show()

#%% Iteración
columnas = descomposiciones.columns
combinaciones = list(itertools.combinations(columnas, 2))

nuevas_filas = []
for col1, col2 in combinaciones:
    resultado = (((descomposiciones[col1] - 1)**2) + (descomposiciones[col2]**2)) ** 0.5
    nuevas_filas.append(resultado)

nuevo_dataframe = pd.DataFrame(nuevas_filas).transpose()
nuevo_dataframe.columns = [f'({col1}, {col2})' for col1, col2 in combinaciones]

columnas_objetivo = ['CE']

for col_objetivo in columnas_objetivo:
    correlaciones = nuevo_dataframe.corrwith(datos_campo[col_objetivo], method='pearson')
    
    correlaciones_ordenadas = correlaciones.sort_values(ascending=False)
    
    print(f"10 valores más altos para '{col_objetivo}':")
    print(correlaciones_ordenadas.head(10))

    print(f"\n10 valores más bajos para '{col_objetivo}':")
    print(correlaciones_ordenadas.tail(10))
    print('\n')

#%% Iteración
columnas = descomposiciones_y_docker.columns
combinaciones = list(itertools.combinations(columnas, 2))

nuevas_filas = []
for col1, col2 in combinaciones:
    resultado = (((descomposiciones_y_docker[col1] - 1)**2) + (descomposiciones_y_docker[col2]**2)) ** 0.5
    nuevas_filas.append(resultado)

nuevo_dataframe = pd.DataFrame(nuevas_filas).transpose()
nuevo_dataframe.columns = [f'({col1}, {col2})' for col1, col2 in combinaciones]

columnas_objetivo = ['CE']

for col_objetivo in columnas_objetivo:
    correlaciones = nuevo_dataframe.corrwith(datos_campo[col_objetivo], method='pearson')
    
    correlaciones_ordenadas = correlaciones.sort_values(ascending=False)
    
    print(f"10 valores más altos para '{col_objetivo}':")
    print(correlaciones_ordenadas.head(10))

    print(f"\n10 valores más bajos para '{col_objetivo}':")
    print(correlaciones_ordenadas.tail(10))
    print('\n')

#%%
"""
ÓPTICOS Y RADAR
"""
df = pd.concat([descomposiciones_y_docker, indices_opticos], axis=1)

#%%  ANTES
columnas = df.columns
fig, ax = plt.subplots()
df[columnas].boxplot(ax=ax)
ax.set_xticklabels(columnas, rotation=90, ha='right')
plt.xlabel('Índices')
plt.ylabel('Valores')
plt.show()

#%% normalización
scaler = StandardScaler()
columnas = df.columns
df[columnas] = scaler.fit_transform(df[columnas])

#%%  DESPUÉS
columnas = df.columns
fig, ax = plt.subplots()
df[columnas].boxplot(ax=ax)
ax.set_xticklabels(columnas, rotation=90, ha='right')
plt.xlabel('Índices')
plt.ylabel('Valores')
plt.show()

#%%
fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(df["NDVI"], df["PAU_2"], estilo, color=color)
plt.xlabel("NDVI")
plt.ylabel("Pauli_g")
plt.show()

fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(df["SI1"], df["PAU_2"], estilo, color=color)
plt.xlabel("SI1")
plt.ylabel("Pauli_g")
plt.show()

fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(df["SAI5"], df["vanZyl_3"], estilo, color=color)
plt.xlabel("SAI5")
plt.ylabel("vanZyl_3")
plt.show()

#%%
NDVI_PAULIg = pd.DataFrame({'NDVI_PAULIg': ((df['NDVI']**2) + (df['PAU_2']**2)) ** (1/2),
                         'CE': datos_campo['CE']}).dropna()

SI_PAULIg = pd.DataFrame({'SI_PAULIg': (((df['SI1']-1)**2) + (df['PAU_2']**2)) ** (1/2),
                         'CE': datos_campo['CE']}).dropna()

SAI5_vanZyl3 = pd.DataFrame({'SAI5_vanZyl3': (((df['SAI5']-1)**2) + (df['vanZyl_3']**2)) ** (1/2),
                         'CE': datos_campo['CE']}).dropna()

#%% CORRELACIONES - PEARSON Y SPEARMAN

######################  NDVI-Pauli_g ####################################
corr_pearson = NDVI_PAULIg["NDVI_PAULIg"].corr(NDVI_PAULIg["CE"], method="pearson")
corr_spearman = NDVI_PAULIg["NDVI_PAULIg"].corr(NDVI_PAULIg["CE"], method="spearman")
corr_pearson_4 = pearsonr(x = NDVI_PAULIg["NDVI_PAULIg"], y =  NDVI_PAULIg["CE"])
print(f"Correlación de Pearson NDVI_PAULIg-CE: {corr_pearson}")
print(f"Correlación de Spearman NDVI_PAULIg-CE: {corr_spearman}")
print("P-value: ", corr_pearson_4[1])

fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(NDVI_PAULIg["CE"], NDVI_PAULIg["NDVI_PAULIg"], estilo, color=color)
plt.xlabel("CE")
plt.ylabel("NDVI_PAULIg")
plt.show()


######################  SI-Pauli_g ####################################
corr_pearson = SI_PAULIg["SI_PAULIg"].corr(SI_PAULIg["CE"], method="pearson")
corr_spearman = SI_PAULIg["SI_PAULIg"].corr(SI_PAULIg["CE"], method="spearman")
corr_pearson_4 = pearsonr(x = SI_PAULIg["SI_PAULIg"], y =  SI_PAULIg["CE"])
print(f"Correlación de Pearson SI_PAULIg-CE: {corr_pearson}")
print(f"Correlación de Spearman SI_PAULIg-CE: {corr_spearman}")
print("P-value: ", corr_pearson_4[1])

fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(SI_PAULIg["CE"], SI_PAULIg["SI_PAULIg"], estilo, color=color)
plt.xlabel("CE")
plt.ylabel("SI_PAULIg")
plt.show()

######################  SAI5-vanZyl_3 ####################################
corr_pearson = SAI5_vanZyl3["SAI5_vanZyl3"].corr(SAI5_vanZyl3["CE"], method="pearson")
corr_spearman = SAI5_vanZyl3["SAI5_vanZyl3"].corr(SAI5_vanZyl3["CE"], method="spearman")
corr_pearson_4 = pearsonr(x = SAI5_vanZyl3["SAI5_vanZyl3"], y =  SAI5_vanZyl3["CE"])
print(f"Correlación de Pearson SAI5_vanZyl3-CE: {corr_pearson}")
print(f"Correlación de Spearman SAI5_vanZyl3-CE: {corr_spearman}")
print("P-value: ", corr_pearson_4[1])

fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(SAI5_vanZyl3["CE"], SAI5_vanZyl3["SAI5_vanZyl3"], estilo, color=color)
plt.xlabel("CE")
plt.ylabel("SAI5_vanZyl3")
plt.show()

#%%
"""
ITERACIÓN TOTAL (elegir entre escalar o normalizar los datos)
"""
#%% ESCALAR
columnas = df.columns
fig, ax = plt.subplots()
df[columnas].boxplot(ax=ax)
ax.set_xticklabels(columnas, rotation=90, ha='right')
plt.xlabel('Índices')
plt.ylabel('Valores')
plt.show()

scaler = StandardScaler()
columnas = df.columns
df[columnas] = scaler.fit_transform(df[columnas])

columnas = df.columns
fig, ax = plt.subplots()
df[columnas].boxplot(ax=ax)
ax.set_xticklabels(columnas, rotation=90, ha='right')
plt.xlabel('Índices')
plt.ylabel('Valores')
plt.show()

#%% NORMALIZAR entre 0 y 1
# Configurar el estilo de fuente
font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}
plt.rc('font', **font)

# Crear el gráfico original
columnas = df.columns
fig, ax = plt.subplots()
df[columnas].boxplot(ax=ax)
ax.set_xticklabels(columnas, rotation=90, ha='right')
plt.xlabel('Índices', fontweight='bold')  # Etiqueta x en negrita
plt.ylabel('Valores', fontweight='bold')  # Etiqueta y en negrita
plt.title('Datos Originales', fontweight='bold', fontsize=16)  # Título en negrita y tamaño de fuente 16
plt.show()

# Normalizar los datos
scaler = MinMaxScaler()
columnas = df.columns
df[columnas] = scaler.fit_transform(df[columnas])

# Crear el gráfico de los datos normalizados
columnas = df.columns
fig, ax = plt.subplots()
df[columnas].boxplot(ax=ax)
ax.set_xticklabels(columnas, rotation=90, ha='right')
plt.xlabel('Índices', fontweight='bold')  # Etiqueta x en negrita
plt.ylabel('Valores', fontweight='bold')  # Etiqueta y en negrita
plt.title('Datos Normalizados de 0 a 1', fontweight='bold', fontsize=16)  # Título en negrita y tamaño de fuente 16
plt.show()

#%%
columnas_df = df.columns
combinaciones = list(itertools.product(columnas_df, repeat=2))

nuevas_filas1 = []
nuevas_filas2 = []
nuevas_filas3 = []
nuevas_filas4 = []
nuevas_filas5 = []
for col1, col2 in combinaciones:
    resultado1 = (((df[col1] - 1)**2) + (df[col2]**2)) ** 0.5
    resultado2 = ((df[col1]**2) + df[col2]) ** 0.5
    resultado3 = (((1 - df[col1])**2) + df[col2]) ** 0.5
    resultado4 = ((df[col1]**2) + (df[col2]**2)) ** 0.5
    resultado5 = (((1 - df[col1])**2) + ((1 - df[col2])**2)) ** 0.5
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
#transformacion = datos_campo['CE']                                              # 0,35    # 0,36
#transformacion = datos_campo['log_CE'] = np.log(datos_campo['CE'])              # 0,40    # 0,40
transformacion = datos_campo['sqrt_CE'] = datos_campo['CE']**2                  # 0,33    # 0,33
#transformacion = datos_campo['raiz_cuadrada_CE'] = np.sqrt(datos_campo['CE'])   # 0,38    # 0,38
#transformacion = datos_campo['inverso_CE'] = 1 / datos_campo['CE']              # 0,39    # 0,39
#transformacion = datos_campo['exp_CE'] = np.exp(datos_campo['CE'])              # 0,26    # 0,32
#transformacion = datos_campo['abs_CE'] = np.abs(datos_campo['CE'])              # 0,36    # 0,36
#transformacion = datos_campo['round_CE'] = np.round(datos_campo['CE'])          # 0,36    # 0,36
#transformacion = datos_campo['CE_elevado_a_3'] = datos_campo['CE'] ** 3         # 0,33    # 0,33



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

    
    print(f"(((df[col1] - 1)**2) + (df[col2]**2)) ** 0.5 / 10 valores más altos para '{col_objetivo}':")
    print(correlaciones_ordenadas1.head(10))
    
    print(f"((df[col1]**2) + df[col2]) ** 0.5 / 10 valores más altos para '{col_objetivo}':")
    print(correlaciones_ordenadas2.head(10))
    
    print(f"(((1 - df[col1])**2) + df[col2]) ** 0.5 / 10 valores más altos para '{col_objetivo}':")
    print(correlaciones_ordenadas3.head(10))
    
    print(f"((df[col1]**2) + (df[col2]**2)) ** 0.5 / 10 valores más altos para '{col_objetivo}':")
    print(correlaciones_ordenadas4.head(10))
    
    print(f"(((1 - df[col1])**2) + ((1 - df[col2])**2)) ** 0.5 / 10 valores más altos para '{col_objetivo}':")
    print(correlaciones_ordenadas5.head(10))
    
#%% 
"""
la mejor correlación fue (((1 - df[col1])**2) + df[col2]) ** 0.5  
(EVI, MNDWI) ---> 0.588206
"""
evi_mndwi = indices_opticos[["EVI", "MNDWI"]]

columnas = indices_opticos.columns
fig, ax = plt.subplots()
indices_opticos[columnas].boxplot(ax=ax)
ax.set_xticklabels(columnas, rotation=90, ha='right')
plt.xlabel('Índices')
plt.ylabel('Valores')
plt.title('Datos Originales')
plt.show()

for columna in indices_opticos.columns:
    valor_minimo = indices_opticos[columna].min()
    valor_maximo = indices_opticos[columna].max()
    indices_opticos[columna] = (indices_opticos[columna] - valor_minimo) / (valor_maximo - valor_minimo)

columnas = indices_opticos.columns
fig, ax = plt.subplots()
indices_opticos[columnas].boxplot(ax=ax)
ax.set_xticklabels(columnas, rotation=90, ha='right')
plt.xlabel('Índices')
plt.ylabel('Valores')
plt.title('Datos Normalizados de 0 a 1')
plt.show()

EVI_MNDWI = indices_opticos[["EVI", "MNDWI"]]

fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(indices_opticos["EVI"], indices_opticos["MNDWI"], estilo, color=color)
plt.xlabel("EVI")
plt.ylabel("MNDWI")
plt.show()

log_CE = np.log(datos_campo['CE'])

EVI_MNDWI_CE = pd.DataFrame({'EVI_MNDWI': (((1 - indices_opticos['EVI'])**2) + indices_opticos['MNDWI']) ** 0.5,
                         'CE': log_CE}).dropna()

corr_pearson = EVI_MNDWI_CE["EVI_MNDWI"].corr(EVI_MNDWI_CE["CE"], method="pearson")
print(f"Correlación de Pearson EVI_MNDWI_CE: {corr_pearson}")

fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(EVI_MNDWI_CE["CE"], EVI_MNDWI_CE["EVI_MNDWI"], estilo, color=color)
plt.xlabel("log(CE)")
plt.ylabel("Índice Feature Space (EVI_MNDWI)")
plt.show()

#%%
CE = EVI_MNDWI_CE['CE'] #esto en realidad es el log(CE)
EVI_MNDWI = EVI_MNDWI_CE['EVI_MNDWI']

slope, intercept, r_value, p_value, std_err = linregress(CE, EVI_MNDWI)

rango_CE = np.linspace(min(CE), max(CE), 100)
EVI_MNDWI_fit = slope * rango_CE + intercept

fig, ax = plt.subplots()
estilo = "o"
color = "steelblue"
ax.plot(CE, EVI_MNDWI, estilo, color=color)
ax.plot(rango_CE, EVI_MNDWI_fit, color="red", label=f"Regresión lineal: y = {slope:.2f}x + {intercept:.2f}")
plt.xlabel("log(CE)")
plt.ylabel("Índice Feature Space (EVI_MNDWI)")
plt.legend()
plt.show()

formula = f"Regresión lineal: log(CE) = {slope:.2f}(EVI_MNDWI) + {intercept:.2f}"
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










