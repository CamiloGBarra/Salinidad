#%%
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
#from correlaciones_sar_procesados import df_sar

#%%
entropia = r'C:\Users\camil\Downloads\Salinidad\imagenes SALINIDAD\procesadas\PP\EOL1ASARSAO1A6877636_2023_04_06_orbit_DESCENDING\con speckle_refinedlee\ENT.tif'
alpha = r'C:\Users\camil\Downloads\Salinidad\imagenes SALINIDAD\procesadas\PP\EOL1ASARSAO1A6877636_2023_04_06_orbit_DESCENDING\con speckle_refinedlee\ALP.tif'

#%% #imágenes completas
with rasterio.open(entropia) as src1:
    imagen1 = src1.read(1)  # Lee la primera banda de la imagen

with rasterio.open(alpha) as src2:
    imagen2 = src2.read(1)  # Lee la primera banda de la imagen

# valores de los píxeles
valores_imagen1 = imagen1.flatten()
valores_imagen2 = imagen2.flatten()

#%% valores de salinidad de campo
CE = pd.read_csv("C://Users//camil//Downloads//Salinidad//Primera campaña SALINIDAD//Campaña_abril2023//suelo_abr2023.csv", sep=";", decimal=".")

CE['Clasificación_CE'] = pd.cut(CE['CE'], bins=[0, 2, 4, 8, 16, float('inf')], 
                             labels=['No salino', 'Ligeramente salino', 'Moderadamente salino', 
                                     'Fuertemente salino', 'Extremadamente salino'])
CE = CE.drop(columns=[col for col in CE.columns if col not in ['CE', 'Clasificación_CE']])

mapeo = {
    'No salino': 0,
    'Ligeramente salino': 1,
    'Moderadamente salino': 2,
    'Fuertemente salino': 3,
    'Extremadamente salino': 4
}

CE['CE_map'] = CE['Clasificación_CE'].map(mapeo)

#%% #puntos de campo
df_docker = pd.read_csv('C://Users//camil//Downloads//Salinidad//Primera campaña SALINIDAD//Campaña_abril2023//indices_polarimetricos.csv')
df_sar = df_docker.drop(columns=[col for col in df_docker.columns if col not in ["ALP", "ENT"]])
#valores_imagen3 = df_docker['ALP'].values.astype(np.float32)
#valores_imagen4 = df_docker['ENT'].values.astype(np.float32)

CE_ALP_ENT = pd.concat([CE, df_sar], axis=1)

#%%
plt.hexbin(valores_imagen1, valores_imagen2, gridsize=500, cmap='GnBu_r', mincnt=1)
cmap=plt.cm.get_cmap('viridis_r')

plt.scatter(CE_ALP_ENT['ENT'], CE_ALP_ENT['ALP'], s=30, c=CE_ALP_ENT['CE'], cmap=cmap)  # datos de campo

plt.xlabel('Entropía', fontsize=13, fontweight='bold')
plt.ylabel('Alpha', fontsize=13, fontweight='bold')
plt.title('H-Alpha', fontsize=15, fontweight='bold')

plt.xlim(0, 1)
plt.ylim(0, 100)

plt.plot([0, 0.5], [42.5, 42.5], color='black')
plt.plot([0, 0.5], [47.5, 47.5], color='black')
plt.plot([0.5, 0.9], [50, 50], color='black')
plt.plot([0.5, 1], [40, 40], color='black')
plt.plot([0.9, 1], [55, 55], color='black')
plt.axvline(x=0.5, color='black')
plt.axvline(x=0.9, color='black')

plt.text(1.1, 100, 'A: Superficie. Entropía baja', fontsize=10, fontweight='bold')
plt.text(1.1, 93, 'B: Superficie. Entropía media', fontsize=10, fontweight='bold')
plt.text(1.1, 86, 'C: Superficie. Entropía alta', fontsize=10, fontweight='bold')
plt.text(1.1, 79, 'D: Volumen. Entropía baja', fontsize=10, fontweight='bold')
plt.text(1.1, 72, 'E: Volumen. Entropía media', fontsize=10, fontweight='bold')
plt.text(1.1, 65, 'F: Volumen. Entropía alta', fontsize=10, fontweight='bold')
plt.text(1.1, 58, 'G: Doble rebote. Entropía baja', fontsize=10, fontweight='bold')
plt.text(1.1, 51, 'H: Doble rebote. Entropía media', fontsize=10, fontweight='bold')
plt.text(1.1, 44, 'I: Doble rebote. Entropía alta', fontsize=10, fontweight='bold')

plt.text(1.1, 30, 'Puntos de muestreo', color='black', fontsize=11, fontweight='bold')
plt.text(1.1, 23, '•', color='#440154', fontsize=18, fontweight='bold')
plt.text(1.15, 23, 'Extremadamente salino', fontsize=10, fontweight='bold')
plt.text(1.1, 16, '•', color='#3b528b', fontsize=18, fontweight='bold')
plt.text(1.15, 16, 'Fuertemente salino', fontsize=10, fontweight='bold')
plt.text(1.1, 9, '•', color='#21918c', fontsize=18, fontweight='bold')
plt.text(1.15, 9, 'Moderadamente salino', fontsize=10, fontweight='bold')
plt.text(1.1, 2, '•', color='#5ec962', fontsize=18, fontweight='bold')
plt.text(1.15, 2, 'Ligeramente salino', fontsize=10, fontweight='bold')
plt.text(1.1, -5, '•', color='#fde725', fontsize=18, fontweight='bold')
plt.text(1.15, -5, 'No salino', fontsize=10, fontweight='bold')

plt.text(0.01, 33, 'A', fontsize=15, fontweight='bold')
plt.text(0.8, 7, 'B', fontsize=15, fontweight='bold')
plt.text(0.95, 30, 'C', fontsize=15, fontweight='bold')
plt.text(0.01, 43, 'D', fontsize=12.5, fontweight='bold')
plt.text(0.8, 43, 'E', fontsize=15, fontweight='bold')
plt.text(0.95, 43, 'F', fontsize=15, fontweight='bold')
plt.text(0.01, 90, 'G', fontsize=15, fontweight='bold')
plt.text(0.8, 90, 'H', fontsize=15, fontweight='bold')
plt.text(0.95, 90, 'I', fontsize=15, fontweight='bold')

plt.show()
