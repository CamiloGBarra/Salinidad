#%%
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import seaborn as sns

#%%
imagen_sentinel = 'C://Users//camil//Downloads//Salinidad//imagenes SALINIDAD//Sentinel2//Sentinel2_5abr2023.tif'
nombres_bandas = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b8a', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14']
with rasterio.open(imagen_sentinel) as src:
    transformacion = src.transform
    bandas = src.read()

#%%
ruta_puntos = 'C://Users//camil//Downloads//Salinidad//python_salinidad//shapes//puntos_abr2023.shp'
puntos = gpd.read_file(ruta_puntos)

valores_indices = []
for punto in puntos.geometry:
    coordenada = (punto.x, punto.y)
    fila, columna = rasterio.transform.rowcol(transformacion, coordenada[0], coordenada[1])
    valores_punto = [banda[fila, columna] for banda in bandas]
    valores_indices.append(valores_punto)
    
#%% convertir los valores de los índices en un array numpy
valores_indices = np.array(valores_indices)

#%% por si se quiere imprimir
for i, indice in enumerate(nombres_bandas):
    print(f'Valores del índice {indice}:')
    print(valores_indices[:, i])
    print()
    
#%%
ce = puntos['CE']
df_bandas = pd.DataFrame({'CE': ce})
for i, indice in enumerate(nombres_bandas):
    df_bandas[indice] = valores_indices[:, i]
print(df_bandas)

df_bandas = df_bandas.drop(['b13', 'b14'], axis=1) #elimino las últimas dos columnas
#%%
# matriz de correlación
correlaciones = df_bandas.corr(method='pearson')
mask = np.triu(np.ones_like(correlaciones, dtype=bool))
cmap = sns.diverging_palette(250, 10, as_cmap=True)
plt.figure(figsize=(10, 8))
sns.heatmap(correlaciones, annot=True, cmap=cmap, mask=mask, center=0, fmt=".2f", linewidths=0.5)
plt.title('Correlación, método Pearson')
plt.show()

#%%
for columna in correlaciones.columns:
    if columna != 'CE':
        plt.figure(figsize=(6, 4))
        plt.scatter(df_bandas['CE'], df_bandas[columna], alpha=0.7)
        plt.xlabel('CE')
        plt.ylabel(columna)
        plt.title(f'Gráfico de dispersión entre CE y {columna}')
        plt.show()
        
#%%


