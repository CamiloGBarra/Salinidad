import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

#%%
humedad = r"C:\Users\camil\Downloads\Salinidad\Mapa humedad\humedad.csv"
humedad = pd.read_csv(humedad)
humedad = humedad * 100

EOL1ASARSAO1A6877636_20230406 = humedad[['EOL1ASARSAO1A6877636_20230406_Humedad_Suelo_DES_ARIDO01', 
                                                           'EOL1ASARSAO1A6877636_20230406_Humedad_Suelo_DES_OH005']]
EOL1ASARSAO1B6869995_20230414 = humedad[['EOL1ASARSAO1B6869995_20230414_Humedad_Suelo_DES_ARIDO01', 
                                                           'EOL1ASARSAO1B6869995_20230414_Humedad_Suelo_DES_OH005']]

EOL1ASARSAO1A6877636_20230406.rename(columns={'EOL1ASARSAO1A6877636_20230406_Humedad_Suelo_DES_ARIDO01': 'ARIDO'}, inplace=True)
EOL1ASARSAO1A6877636_20230406.rename(columns={'EOL1ASARSAO1A6877636_20230406_Humedad_Suelo_DES_OH005': 'OH'}, inplace=True)
EOL1ASARSAO1B6869995_20230414.rename(columns={'EOL1ASARSAO1B6869995_20230414_Humedad_Suelo_DES_ARIDO01': 'ARIDO'}, inplace=True)
EOL1ASARSAO1B6869995_20230414.rename(columns={'EOL1ASARSAO1B6869995_20230414_Humedad_Suelo_DES_OH005': 'OH'}, inplace=True)

#%%
csv_muestreo_abril2023 = r"C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña_abril2023\suelo_abr2023.csv"
csv_muestreo_abril2023 = pd.read_csv(csv_muestreo_abril2023, sep=';', decimal='.')
csv_muestreo_abril2023 = csv_muestreo_abril2023[['Muestra','CE', 'Humedad', 'RAS', 'pH', 'Suelo_desnudo_porc']]
CE = csv_muestreo_abril2023[['CE']]

#%% suelo desnudo
df1 = pd.merge(EOL1ASARSAO1A6877636_20230406, csv_muestreo_abril2023, left_index=True, right_index=True, how='inner')
df2 = pd.merge(EOL1ASARSAO1A6877636_20230406, csv_muestreo_abril2023, left_index=True, right_index=True, how='inner')
df3 = pd.merge(EOL1ASARSAO1B6869995_20230414, csv_muestreo_abril2023, left_index=True, right_index=True, how='inner')
df4 = pd.merge(EOL1ASARSAO1B6869995_20230414, csv_muestreo_abril2023, left_index=True, right_index=True, how='inner')

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Gráfico 1
sns.scatterplot(data=df1, x="Humedad", y="ARIDO", hue="Suelo_desnudo_porc", palette="viridis", ax=axes[0, 0])
axes[0, 0].plot([0, 100], [0, 100], color='red', linestyle='--')
axes[0, 0].set_xlim(0, 50)
axes[0, 0].set_ylim(0, 60)
axes[0, 0].set_title("EOL1ASARSAO1A6877636_20230406")
axes[0, 0].set_xlabel("Humedad gravimétrica")
axes[0, 0].set_ylabel("Humedad (modelo ARIDO)")
axes[0, 0].legend(title="Suelo desnudo")

# Gráfico 2
sns.scatterplot(data=df2, x="Humedad", y="OH", hue="Suelo_desnudo_porc", palette="viridis", ax=axes[0, 1])
axes[0, 1].plot([0, 100], [0, 100], color='red', linestyle='--')
axes[0, 1].set_xlim(0, 50)
axes[0, 1].set_ylim(0, 60)
axes[0, 1].set_title("EOL1ASARSAO1A6877636_20230406")
axes[0, 1].set_xlabel("Humedad gravimétrica")
axes[0, 1].set_ylabel("Humedad (modelo OH)")
axes[0, 1].legend(title="Suelo desnudo")

# Gráfico 3
sns.scatterplot(data=df3, x="Humedad", y="ARIDO", hue="Suelo_desnudo_porc", palette="viridis", ax=axes[1, 0])
axes[1, 0].plot([0, 100], [0, 100], color='red', linestyle='--')
axes[1, 0].set_xlim(0, 50)
axes[1, 0].set_ylim(0, 60)
axes[1, 0].set_title("EOL1ASARSAO1B6869995_20230414")
axes[1, 0].set_xlabel("Humedad gravimétrica")
axes[1, 0].set_ylabel("Humedad (modelo ARIDO)")
axes[1, 0].legend(title="Suelo desnudo")

# Gráfico 4
sns.scatterplot(data=df4, x="Humedad", y="OH", hue="Suelo_desnudo_porc", palette="viridis", ax=axes[1, 1])
axes[1, 1].plot([0, 100], [0, 100], color='red', linestyle='--')
axes[1, 1].set_xlim(0, 50)
axes[1, 1].set_ylim(0, 60)
axes[1, 1].set_title("EOL1ASARSAO1B6869995_20230414")
axes[1, 1].set_xlabel("Humedad gravimétrica")
axes[1, 1].set_ylabel("Humedad (modelo OH)")
axes[1, 1].legend(title="Suelo desnudo")

plt.tight_layout()
plt.show()

#%% salinidad
df1 = pd.merge(EOL1ASARSAO1A6877636_20230406, csv_muestreo_abril2023, left_index=True, right_index=True, how='inner')
df2 = pd.merge(EOL1ASARSAO1A6877636_20230406, csv_muestreo_abril2023, left_index=True, right_index=True, how='inner')
df3 = pd.merge(EOL1ASARSAO1B6869995_20230414, csv_muestreo_abril2023, left_index=True, right_index=True, how='inner')
df4 = pd.merge(EOL1ASARSAO1B6869995_20230414, csv_muestreo_abril2023, left_index=True, right_index=True, how='inner')

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Gráfico 1
sns.scatterplot(data=df1, x="Humedad", y="ARIDO", hue="CE", palette="viridis", ax=axes[0, 0])
axes[0, 0].plot([0, 100], [0, 100], color='red', linestyle='--')
axes[0, 0].set_xlim(0, 50)
axes[0, 0].set_ylim(0, 60)
axes[0, 0].set_title("EOL1ASARSAO1A6877636_20230406")
axes[0, 0].set_xlabel("Humedad gravimétrica")
axes[0, 0].set_ylabel("Humedad (modelo ARIDO)")
axes[0, 0].legend(title="CE")

# Gráfico 2
sns.scatterplot(data=df2, x="Humedad", y="OH", hue="CE", palette="viridis", ax=axes[0, 1])
axes[0, 1].plot([0, 100], [0, 100], color='red', linestyle='--')
axes[0, 1].set_xlim(0, 50)
axes[0, 1].set_ylim(0, 60)
axes[0, 1].set_title("EOL1ASARSAO1A6877636_20230406")
axes[0, 1].set_xlabel("Humedad gravimétrica")
axes[0, 1].set_ylabel("Humedad (modelo OH)")
axes[0, 1].legend(title="CE")

# Gráfico 3
sns.scatterplot(data=df3, x="Humedad", y="ARIDO", hue="CE", palette="viridis", ax=axes[1, 0])
axes[1, 0].plot([0, 100], [0, 100], color='red', linestyle='--')
axes[1, 0].set_xlim(0, 50)
axes[1, 0].set_ylim(0, 60)
axes[1, 0].set_title("EOL1ASARSAO1B6869995_20230414")
axes[1, 0].set_xlabel("Humedad gravimétrica")
axes[1, 0].set_ylabel("Humedad (modelo ARIDO)")
axes[1, 0].legend(title="CE")

# Gráfico 4
sns.scatterplot(data=df4, x="Humedad", y="OH", hue="CE", palette="viridis", ax=axes[1, 1])
axes[1, 1].plot([0, 100], [0, 100], color='red', linestyle='--')
axes[1, 1].set_xlim(0, 50)
axes[1, 1].set_ylim(0, 60)
axes[1, 1].set_title("EOL1ASARSAO1B6869995_20230414")
axes[1, 1].set_xlabel("Humedad gravimétrica")
axes[1, 1].set_ylabel("Humedad (modelo OH)")
axes[1, 1].legend(title="CE")

plt.tight_layout()
plt.show()

#%% pH
df1 = pd.merge(EOL1ASARSAO1A6877636_20230406, csv_muestreo_abril2023, left_index=True, right_index=True, how='inner')
df2 = pd.merge(EOL1ASARSAO1A6877636_20230406, csv_muestreo_abril2023, left_index=True, right_index=True, how='inner')
df3 = pd.merge(EOL1ASARSAO1B6869995_20230414, csv_muestreo_abril2023, left_index=True, right_index=True, how='inner')
df4 = pd.merge(EOL1ASARSAO1B6869995_20230414, csv_muestreo_abril2023, left_index=True, right_index=True, how='inner')

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Gráfico 1
sns.scatterplot(data=df1, x="Humedad", y="ARIDO", hue="pH", palette="viridis", ax=axes[0, 0])
axes[0, 0].plot([0, 100], [0, 100], color='red', linestyle='--')
axes[0, 0].set_xlim(0, 50)
axes[0, 0].set_ylim(0, 60)
axes[0, 0].set_title("EOL1ASARSAO1A6877636_20230406")
axes[0, 0].set_xlabel("Humedad gravimétrica")
axes[0, 0].set_ylabel("Humedad (modelo ARIDO)")
axes[0, 0].legend(title="pH")

# Gráfico 2
sns.scatterplot(data=df2, x="Humedad", y="OH", hue="pH", palette="viridis", ax=axes[0, 1])
axes[0, 1].plot([0, 100], [0, 100], color='red', linestyle='--')
axes[0, 1].set_xlim(0, 50)
axes[0, 1].set_ylim(0, 60)
axes[0, 1].set_title("EOL1ASARSAO1A6877636_20230406")
axes[0, 1].set_xlabel("Humedad gravimétrica")
axes[0, 1].set_ylabel("Humedad (modelo OH)")
axes[0, 1].legend(title="pH")

# Gráfico 3
sns.scatterplot(data=df3, x="Humedad", y="ARIDO", hue="pH", palette="viridis", ax=axes[1, 0])
axes[1, 0].plot([0, 100], [0, 100], color='red', linestyle='--')
axes[1, 0].set_xlim(0, 50)
axes[1, 0].set_ylim(0, 60)
axes[1, 0].set_title("EOL1ASARSAO1B6869995_20230414")
axes[1, 0].set_xlabel("Humedad gravimétrica")
axes[1, 0].set_ylabel("Humedad (modelo ARIDO)")
axes[1, 0].legend(title="pH")

# Gráfico 4
sns.scatterplot(data=df4, x="Humedad", y="OH", hue="pH", palette="viridis", ax=axes[1, 1])
axes[1, 1].plot([0, 100], [0, 100], color='red', linestyle='--')
axes[1, 1].set_xlim(0, 50)
axes[1, 1].set_ylim(0, 60)
axes[1, 1].set_title("EOL1ASARSAO1B6869995_20230414")
axes[1, 1].set_xlabel("Humedad gravimétrica")
axes[1, 1].set_ylabel("Humedad (modelo OH)")
axes[1, 1].legend(title="pH")

plt.tight_layout()
plt.show()

