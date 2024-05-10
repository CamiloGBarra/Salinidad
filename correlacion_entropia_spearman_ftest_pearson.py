#%%
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt

#%%
"""
DATOS
"""
#%%
csv_muestreo_abril2023 = r"C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\suelo_abr2023.csv"
csv_muestreo_abril2023 = pd.read_csv(csv_muestreo_abril2023, sep=';', decimal='.')
#csv_muestreo_abril2023 = csv_muestreo_abril2023[['Muestra','CE', 'Humedad', 'RAS', 'pH', 'Suelo_desnudo_porc']]
CE = csv_muestreo_abril2023[['CE']]

#%%
opticos = r"C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\indices_opticos_5_abr.csv"
opticos = pd.read_csv(opticos)
print("Columnas con valores NaN:", opticos.columns[opticos.isna().any()].tolist())

polarimetria = r"C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\indices_polarimetricos.csv"
polarimetria = pd.read_csv(polarimetria)
print("Columnas con valores NaN:", polarimetria.columns[polarimetria.isna().any()].tolist())

dataset = pd.concat([opticos, 
                     polarimetria
                     ], 
                    axis=1)

#%%
df = pd.concat([CE, dataset], axis=1)

#%%
resultados_correlacion = []

for columna in df.columns:
    if columna != 'CE':
        # F-test
        f_test_score, _ = scipy.stats.f_oneway(df[columna], df['CE'])

        # Pearson
        pearson_corr, _ = scipy.stats.pearsonr(df[columna], df['CE'])

        # Spearman
        spearman_corr, _ = scipy.stats.spearmanr(df[columna], df['CE'])

        # Entropía
        entropia_corr = mutual_info_regression(df[[columna]], df['CE'])[0]

        resultados_correlacion.append({
            'Columna': columna,
            'F-test': f_test_score,
            'Pearson': pearson_corr,
            'Spearman': spearman_corr,
            'Entropía': entropia_corr
        })

resultados_df = pd.DataFrame(resultados_correlacion)

print(resultados_df)

#%%
# Eliminar la columna 'Columna' para evitar representarla en el gráfico
resultados_df = resultados_df.set_index('Columna')

# Crear un mapa de calor de las correlaciones
plt.figure(figsize=(10, 8))
sns.heatmap(resultados_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlaciones entre CE y otras columnas')
plt.show()





