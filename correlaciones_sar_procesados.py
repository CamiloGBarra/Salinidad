#%%
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

#%%
ce_df = pd.read_csv('C://Users//camil//Downloads//Salinidad//Primera campaña SALINIDAD//Campaña_abril2023//suelo_abr2023.csv', sep=';', decimal='.')
indices_polarimetricos = pd.read_csv(r'C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\indices_polarimetricos.csv')
bandas_saocom = pd.read_csv(r'C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\saocom_quadpol.csv')

otros_df = pd.concat([indices_polarimetricos, bandas_saocom], axis=1)

df_sar = pd.concat([ce_df['CE'], otros_df], axis=1)
df_sar = df_sar.dropna()

df_sar_noce = df_sar.drop(columns=['CE'])

#%%
# primero obtener las columnas del df excepto 'CE'
columnas = df_sar.columns.tolist()
columnas.remove('CE')

# se crea una lista para almacenar los resultados
resultados_spearman = []
resultados_pearson = []

# correlaciones
for columna in columnas:
    # Spearman
    coef_spearman, _ = spearmanr(df_sar['CE'], df_sar[columna])
    resultados_spearman.append(coef_spearman)
    
    # Pearson
    coef_pearson, _ = pearsonr(df_sar['CE'], df_sar[columna])
    resultados_pearson.append(coef_pearson)

# crear un nuevo df con los resultados
resultados_df = pd.DataFrame({'Columna': columnas,
                              'Correlación Spearman': resultados_spearman,
                              'Correlación Pearson': resultados_pearson})
print(resultados_df)

#%%
correlation_matrix = df_sar.corr()

# máscara para la mitad superior
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# estilo
sns.set(style="white")
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# figura y ejes
fig, ax = plt.subplots(figsize=(10, 8))

# mapa de calor con la figura geométrica en una mitad
sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=False, fmt=".2f")

# ajustar la posición de las etiquetas en el eje x
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, horizontalalignment='right')

# gráfico
plt.title("Matriz de Correlación")
plt.show()

#%%
ce_correlations = df_sar.corr()['CE'].drop('CE')

df_ce_correlations = pd.DataFrame({
    'Columnas': ce_correlations.index,
    'Correlación': ce_correlations.values
})

# Ordenar el DataFrame por correlación de mayor a menor
df_ce_correlations = df_ce_correlations.sort_values(by='Correlación', ascending=False)

sns.set(style="whitegrid")
plt.figure(figsize=(12, 10))
pal = sns.color_palette("coolwarm", len(ce_correlations))
rank = df_ce_correlations['Correlación'].argsort().argsort()  # Usamos el orden del DataFrame ordenado
barplot = sns.barplot(x='Columnas', y='Correlación', data=df_ce_correlations, palette=np.array(pal[::-1])[rank])
plt.title('Correlación CE')
plt.xlabel('Columnas')
plt.ylabel('Correlación')
plt.xticks(rotation=90)

# Establecer límites en el eje y
plt.ylim(-1, 1)

for i, v in enumerate(df_ce_correlations['Correlación']):
    barplot.text(i, v, f"{v:.2f}", ha='center', va='top', fontsize=12, rotation=90)

plt.tight_layout()
plt.show()