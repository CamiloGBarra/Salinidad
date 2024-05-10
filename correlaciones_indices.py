#%%
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import seaborn as sns
from tabulate import tabulate
from colorama import Fore, Back, Style
import matplotlib.font_manager as fm

#%%"
imagen_sentinel_5abr = r'C:\Users\camil\Downloads\Salinidad\imagenes SALINIDAD\Sentinel-2_RGB\5_abr.tif'
with rasterio.open(imagen_sentinel_5abr) as src:
    transformacion = src.transform
    bandas = src.read()

"""imagen_sentinel_anual = 'C://Users//camil//Downloads//Salinidad//imagenes SALINIDAD//Sentinel2//Sentinel2_mediaAnual_2022abr-2023abr.tif'
with rasterio.open(imagen_sentinel_anual) as src:
    transformacion = src.transform
    bandas = src.read()"""
#%%
ruta_puntos = 'C://Users//camil//Downloads//Salinidad//Primera campaña SALINIDAD//Campaña_abril2023//puntos.shp'
puntos = gpd.read_file(ruta_puntos)

valores_indices = []
for punto in puntos.geometry:
    coordenada = (punto.x, punto.y)
    fila, columna = rasterio.transform.rowcol(transformacion, coordenada[0], coordenada[1])
    ndvi = (bandas[8, fila, columna] - bandas[4, fila, columna]) / (bandas[8, fila, columna] + bandas[4, fila, columna])
    ndsi = (bandas[4, fila, columna] - bandas[8, fila, columna]) / (bandas[4, fila, columna] + bandas[8, fila, columna])
    mndwi = (bandas[3, fila, columna] - bandas[11, fila, columna]) / (bandas[3, fila, columna] + bandas[11, fila, columna])
    evi = 2.5 * ((bandas[8, fila, columna] - bandas[4, fila, columna]) / (6 * bandas[4, fila, columna] - 7.5 * bandas[2, fila, columna] + bandas[8, fila, columna] + 1))
    savi = 1.5 * ((bandas[8, fila, columna] - bandas[4, fila, columna]) / (bandas[8, fila, columna] + bandas[4, fila, columna] + 0.5))
    osavi = 1.16 * ((bandas[8, fila, columna] - bandas[4, fila, columna]) / (bandas[8, fila, columna] + bandas[4, fila, columna] + 0.16))
    crsi = (((bandas[8, fila, columna] * bandas[4, fila, columna]) - (bandas[3, fila, columna] * bandas[2, fila, columna])) / ((bandas[8, fila, columna] * bandas[4, fila, columna]) + (bandas[3, fila, columna] * bandas[2, fila, columna]))) ** (0.5)
    khaier = (bandas[8, fila, columna] - bandas[12, fila, columna]) / (bandas[8, fila, columna] + bandas[12, fila, columna])
    gari = (bandas[8, fila, columna] - (0.9 * (bandas[2, fila, columna] - bandas[4, fila, columna]) + bandas[3, fila, columna])) / (bandas[8, fila, columna] + (0.9 * (bandas[2, fila, columna] - bandas[4, fila, columna]) + bandas[3, fila, columna]))
    gdvi2 = ((bandas[8, fila, columna] ** 2) - (bandas[4, fila, columna] ** 2)) / ((bandas[8, fila, columna] ** 2) + (bandas[4, fila, columna] ** 2))
    bi = ((bandas[8, fila, columna] ** 2) + (bandas[4, fila, columna] ** 2)) ** (0.5)
    ndi = (bandas[12, fila, columna] - bandas[7, fila, columna]) / (bandas[12, fila, columna] + bandas[7, fila, columna])
    tbi = (bandas[12, fila, columna] - bandas[3, fila, columna]) / (bandas[3, fila, columna] - bandas[11, fila, columna])
    si1 = ((bandas[2, fila, columna] * bandas[4, fila, columna])) ** (0.5)
    si2 = ((bandas[3, fila, columna] * bandas[4, fila, columna])) ** (0.5)
    si3 = ((bandas[3, fila, columna] ** 2) + (bandas[4, fila, columna] ** 2) + (bandas[8, fila, columna] ** 2)) ** (0.5)
    si4 = ((bandas[3, fila, columna] + bandas[4, fila, columna])) ** (0.5)
    sai1 = bandas[2, fila, columna] / bandas[4, fila, columna]
    sai2 = (bandas[2, fila, columna] - bandas[4, fila, columna]) / (bandas[2, fila, columna] + bandas[4, fila, columna])
    sai3 = (bandas[3, fila, columna] * bandas[4, fila, columna]) / bandas[2, fila, columna]
    sai4 = (bandas[2, fila, columna] * bandas[4, fila, columna]) / bandas[3, fila, columna]
    sai5 = (bandas[4, fila, columna] * bandas[8, fila, columna]) / bandas[3, fila, columna]
    sai6 = (bandas[3, fila, columna] + bandas[4, fila, columna] + bandas[8, fila, columna]) / 2
    sai7 = (bandas[3, fila, columna] + bandas[4, fila, columna]) / 2
    sai8 = (bandas[11, fila, columna] - bandas[12, fila, columna]) / (bandas[11, fila, columna] + bandas[12, fila, columna])
    sai9 = bandas[11, fila, columna] / bandas[12, fila, columna]
    corsi = ((bandas[2, fila, columna] + bandas[3, fila, columna]) / (bandas[4, fila, columna] + bandas[8, fila, columna])) * ((bandas[4, fila, columna] - bandas[8, fila, columna]) / (bandas[4, fila, columna] + bandas[8, fila, columna]))
    vssi = 2 * bandas[3, fila, columna] - (bandas[4, fila, columna] + bandas[8, fila, columna])
    bsi = ((bandas[4, fila, columna] + bandas[3, fila, columna]) - (bandas[4, fila, columna] + bandas[2, fila, columna])) / ((bandas[4, fila, columna] + bandas[3, fila, columna]) + (bandas[4, fila, columna] + bandas[2, fila, columna]))
    valores_indices.append([ndvi, ndsi, mndwi, evi, savi, osavi, crsi, khaier, gari, gdvi2, bi, ndi, tbi, si1, si2, si3,
                            si4, sai1, sai2, sai3, sai4, sai5, sai6, sai7, sai8, sai9, corsi, vssi, bsi])

# convertir los valores de los índices en un array numpy
valores_indices = np.array(valores_indices)

#%%
ce = puntos['CE']
df_indices = pd.DataFrame({'CE': ce})
df_indices['NDVI'] = valores_indices[:, 0]
df_indices['NDSI'] = valores_indices[:, 1]
df_indices['MNDWI'] = valores_indices[:, 2]
df_indices['EVI'] = valores_indices[:, 3]
df_indices['SAVI'] = valores_indices[:, 4]
df_indices['OSAVI'] = valores_indices[:, 5]
df_indices['CRSI'] = valores_indices[:, 6]
df_indices['KHAIER'] = valores_indices[:, 7]
df_indices['GARI'] = valores_indices[:, 8]
df_indices['GDVI2'] = valores_indices[:, 9]
df_indices['BI'] = valores_indices[:, 10]
df_indices['NDI'] = valores_indices[:, 11]
df_indices['TBI'] = valores_indices[:, 12]
df_indices['SI1'] = valores_indices[:, 13]
df_indices['SI2'] = valores_indices[:, 14]
df_indices['SI3'] = valores_indices[:, 15]
df_indices['SI4'] = valores_indices[:, 16]
df_indices['SAI1'] = valores_indices[:, 17]
df_indices['SAI2'] = valores_indices[:, 18]
df_indices['SAI3'] = valores_indices[:, 19]
df_indices['SAI4'] = valores_indices[:, 20]
df_indices['SAI5'] = valores_indices[:, 21]
df_indices['SAI6'] = valores_indices[:, 22]
df_indices['SAI7'] = valores_indices[:, 23]
df_indices['SAI8'] = valores_indices[:, 24]
df_indices['SAI9'] = valores_indices[:, 25]
df_indices['CORSI'] = valores_indices[:, 26]
df_indices['VSSI'] = valores_indices[:, 27]
df_indices['BSI'] = valores_indices[:, 28]
print(df_indices)

# matriz de correlación
correlation_matrix = df_indices.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
correlation_matrix = correlation_matrix.mask(mask)
ce_correlation = correlation_matrix["CE"]
sns.set(font_scale=1.5)
sns.set_style("whitegrid", {'axes.labelweight': 'bold'})
plt.figure(figsize=(12, 12))
sns.heatmap(correlation_matrix, annot=False, cmap="RdBu", fmt=".2f")
plt.xticks(fontweight="bold")
plt.yticks(fontweight="bold")
cax = plt.gcf().axes[-1]
plt.show()

#%%
ce_correlacion = correlaciones['CE']
ce_correlacion = ce_correlacion.drop('CE')
ce_correlacion_sorted = ce_correlacion.sort_values(ascending=False)
color_palette = sns.color_palette("coolwarm", len(ce_correlacion_sorted))
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=ce_correlacion_sorted.index, y=ce_correlacion_sorted.values, palette=color_palette)

plt.title("Correlación CE con índices espectrales", fontsize=24, fontweight='bold')  # Aumentar el tamaño y poner en negrita el título
plt.ylim(-1, 1)
plt.xlabel("Índices", fontsize=24, fontweight='bold')  # Aumentar el tamaño y poner en negrita la etiqueta x
plt.ylabel("Coeficiente de correlación", fontsize=24, fontweight='bold')  # Aumentar el tamaño y poner en negrita la etiqueta y
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=22, fontweight='bold')  # Aumentar el tamaño de las etiquetas del eje x
ax.set_yticklabels(ax.get_yticklabels(), fontsize=22, fontweight='bold')  # Aumentar el tamaño de las etiquetas del eje y

plt.show()

#%%
tabla_correlacion = pd.DataFrame({'Columna': ce_correlacion_sorted.index, 'Correlación': ce_correlacion_sorted.values})
table_style = tabla_correlacion.style.background_gradient(cmap='coolwarm').format({'Correlación': "{:.2f}"})
table_style
print(tabulate(tabla_correlacion, headers='keys', tablefmt='fancy_grid'))

#%%
tabla_correlacion = tabla_correlacion.sort_values(by="Correlación", ascending=False)

# Crear un mapa de calor utilizando Seaborn
sns.set(style="whitegrid")  # Establecer el estilo del gráfico
plt.figure(figsize=(0.5, 6))  # Tamaño del gráfico
sns.heatmap(tabla_correlacion.set_index("Columna"), cmap="RdBu")  # Crear el mapa de calor

#%%
print(tabla_correlacion)
