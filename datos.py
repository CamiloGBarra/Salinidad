#%%
import pandas as pd
from scipy.stats import describe
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fitter import Fitter, get_common_distributions
from scipy.stats import powerlaw

#%%
df = pd.read_csv("C://Users//camil//Downloads//Salinidad//Primera campaña SALINIDAD//Campaña_abril2023//suelo_abr2023.csv", sep=";", decimal=".")

df['Clasificación_CE'] = pd.cut(df['CE'], bins=[0, 2, 4, 8, 16, float('inf')], 
                             labels=['No salino', 'Ligeramente salino', 'Moderadamente salino', 
                                     'Fuertemente salino', 'Extremadamente salino'])

df['Clasificación_RAS'] = pd.cut(df['RAS'], bins=[0, 5, 13, 18, 28, float('inf')], 
                             labels=['No sódico', 'Ligeramente sódico', 'Moderadamente sódico', 
                                     'Fuertemente sódico', 'Extremadamente sódico'])

df.loc[(df['CE'] > 4) & (df['RAS'] < 12) & (df['pH'] < 8.5), 'Clasificación_Suelo'] = 'Salino'
df.loc[(df['CE'] < 4) & (df['RAS'] > 12) & (df['pH'] > 8.5), 'Clasificación_Suelo'] = 'Sódico'
df.loc[(df['CE'] > 4) & (df['RAS'] > 12) & (df['pH'] < 8.5), 'Clasificación_Suelo'] = 'Salino sódico'
df.loc[(df['CE'] > 4) & (df['RAS'] > 12) & (df['pH'] > 8.5), 'Clasificación_Suelo'] = 'Salino sódico'
df.loc[(df['CE'] < 4) & (df['RAS'] < 12) & (df['pH'] < 8.5), 'Clasificación_Suelo'] = 'Normal'

mapeo = {
    'No salino': 0,
    'Ligeramente salino': 1,
    'Moderadamente salino': 2,
    'Fuertemente salino': 3,
    'Extremadamente salino': 4
}

df['CE_map'] = df['Clasificación_CE'].map(mapeo)

#%%
"""
SALINIDAD
"""
#%%
media = df['CE'].mean()
maximo = df['CE'].max()
minimo = df['CE'].min()
moda = df['CE'].mode()[0]
mediana = df['CE'].median()

print(f"Media: {media}")
print(f"Máximo: {maximo}")
print(f"Mínimo: {minimo}")
print(f"Moda: {moda}")
print(f"Mediana: {mediana}")

#%%
estadisticos = describe(df['CE'])
tabla_estadisticos = pd.DataFrame(estadisticos._asdict())
print(tabla_estadisticos)

#%%
media1 = df['CE'].mean()
maximo1 = df['CE'].max()
minimo1 = df['CE'].min()
moda1 = df['CE'].mode()[0]
mediana1 = df['CE'].median()
std1 = df['CE'].std()
percentiles1 = np.percentile(df['CE'], [25, 50, 75])

tabla_estadisticos1 = pd.DataFrame({
    'Media': [media1],
    'Máximo': [maximo1],
    'Mínimo': [minimo1],
    'Moda': [moda1],
    'Mediana': [mediana1],
    'Desviación estándar': [std1],
    'Primer cuartil': [percentiles1[0]],
    'Segundo cuartil (Mediana)': [percentiles1[1]],
    'Tercer cuartil': [percentiles1[2]]
})

print(tabla_estadisticos1)

#%% PLOT
plt.hist(df['CE'], bins= 70, edgecolor='black')
plt.xlabel('Salinidad')
plt.ylabel('Frecuencia')
plt.title('Distribución de la salinidad del Suelo')
plt.grid(True)
plt.show()

#%%
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 6))
sns.kdeplot(
    df.CE,
    fill    = True,
    color   = "blue",
    ax      = axes[0]
)
sns.rugplot(
    df.CE,
    color   = "blue",
    ax      = axes[0]
)
axes[0].set_title("Distribución original", fontsize = 'medium')
axes[0].set_xlabel('CE', fontsize='small') 
axes[0].tick_params(labelsize = 6)

sns.kdeplot(
    np.sqrt(df.CE),
    fill    = True,
    color   = "blue",
    ax      = axes[1]
)
sns.rugplot(
    np.sqrt(df.CE),
    color   = "blue",
    ax      = axes[1]
)
axes[1].set_title("Transformación raíz cuadrada", fontsize = 'medium')
axes[1].set_xlabel('sqrt(CE)', fontsize='small') 
axes[1].tick_params(labelsize = 6)

sns.kdeplot(
    np.log(df.CE),
    fill    = True,
    color   = "blue",
    ax      = axes[2]
)
sns.rugplot(
    np.log(df.CE),
    color   = "blue",
    ax      = axes[2]
)
axes[2].set_title("Transformación logarítmica", fontsize = 'medium')
axes[2].set_xlabel('log(CE)', fontsize='small') 
axes[2].tick_params(labelsize = 6)

fig.tight_layout()

#%%
"""
librería que permite identificar a qué distribución se ajustan mejor los datos (fitter). 
Esta librería permite ajustar cualquiera de las 80 distribuciones implementadas en scipy.
"""
distribuciones = ['cauchy', 'chi2', 'expon',  'exponpow', 'gamma',
                  'norm', 'powerlaw', 'beta', 'logistic']

fitter = Fitter(df.CE, distributions=distribuciones)
fitter.fit()
fitter.summary(Nbest=10, plot=False)

#%%



#%%
tabla_cantidad = df['Clasificación_CE'].value_counts().reset_index()
tabla_cantidad.columns = ['Clasificación_CE', 'Cantidad']
tabla_cantidad['Porcentaje'] = tabla_cantidad['Cantidad'] / tabla_cantidad['Cantidad'].sum() * 100
tabla_cantidad = tabla_cantidad.sort_values('Clasificación_CE')
print(tabla_cantidad)

#%%
"""
SODICIDAD
"""
#%%
media_RAS = df['RAS'].mean()
maximo_RAS = df['RAS'].max()
minimo_RAS = df['RAS'].min()
moda_RAS = df['RAS'].mode()[0]
mediana_RAS = df['RAS'].median()

print(f"Media: {media}")
print(f"Máximo: {maximo}")
print(f"Mínimo: {minimo}")
print(f"Moda: {moda}")
print(f"Mediana: {mediana}")

#%%
estadisticos_RAS = describe(df['RAS'])
tabla_estadisticos_RAS = pd.DataFrame(estadisticos_RAS._asdict())
print(tabla_estadisticos_RAS)

#%%
media1_RAS = df['RAS'].mean()
maximo1_RAS = df['RAS'].max()
minimo1_RAS = df['RAS'].min()
moda1_RAS = df['RAS'].mode()[0]
mediana1_RAS = df['RAS'].median()
std1_RAS = df['RAS'].std()
percentiles1_RAS = np.percentile(df['RAS'], [25, 50, 75])

tabla_estadisticos1_RAS = pd.DataFrame({
    'Media': [media1_RAS],
    'Máximo': [maximo1_RAS],
    'Mínimo': [minimo1_RAS],
    'Moda': [moda1_RAS],
    'Mediana': [mediana1_RAS],
    'Desviación estándar': [std1_RAS],
    'Primer cuartil': [percentiles1_RAS[0]],
    'Segundo cuartil (Mediana)': [percentiles1_RAS[1]],
    'Tercer cuartil': [percentiles1_RAS[2]]
})

print(tabla_estadisticos1_RAS)

#%% PLOT
plt.hist(df['RAS'], bins= 100, edgecolor='black')
plt.xlabel('Sodicidad')
plt.ylabel('Frecuencia')
plt.title('Distribución de la sodicidad del Suelo')
plt.grid(True)
plt.show()

#%%
tabla_cantidad = df['Clasificación_RAS'].value_counts().reset_index()
tabla_cantidad.columns = ['Clasificación_RAS', 'Cantidad']
tabla_cantidad['Porcentaje'] = tabla_cantidad['Cantidad'] / tabla_cantidad['Cantidad'].sum() * 100
tabla_cantidad = tabla_cantidad.sort_values('Clasificación_RAS')
print(tabla_cantidad)

#%% Colores
colores_ce = {
    'No salino': 'blue',
    'Ligeramente salino': 'green',
    'Moderadamente salino': 'yellow',
    'Fuertemente salino': 'orange',
    'Extremadamente salino': 'red'
}

colores_ras = {
    'No sódico': 'blue',
    'Ligeramente sódico': 'green',
    'Moderadamente sódico': 'yellow',
    'Fuertemente sódico': 'orange',
    'Extremadamente sódico': 'red'
}

colores_suelo = {
    'Salino sódico': 'g',
    'Normal': 'y',
    'Salino': 'r',
}

colores_textura = {
    'Arcilloso': 'cornflowerblue',
    'Franco arcilloso': 'red',
    'Franco arcillo limoso': 'lightseagreen',
    'Franco limoso': 'lightcoral',
    'Franco': 'darkgreen',
    'Franco arenoso': 'orange',
    'Arenoso': 'yellow',
}

#%%
"""
RAS vs CE
"""
#%%
df['log_CE'] = np.log10(df['CE'])

for clasificacion, datos in df.groupby('Clasificación_Suelo'):
    plt.scatter(datos['CE'], datos['RAS'], color=colores_suelo[clasificacion], label=clasificacion)

plt.xlabel('CE', fontsize=14, fontweight='bold')
plt.ylabel('RAS', fontsize=14, fontweight='bold')
plt.title('CE vs RAS', fontsize=16)
plt.legend(prop={'size': 12, 'weight': 'bold'})
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.gca().set_facecolor('lightgrey')
plt.grid(True)
plt.show()

for clasificacion, datos in df.groupby('Textura'):
    plt.scatter(datos['CE'], datos['RAS'], color=colores_textura[clasificacion], label=clasificacion)

plt.xlabel('CE', fontsize=14, fontweight='bold')
plt.ylabel('RAS', fontsize=14, fontweight='bold')
plt.title('CE vs RAS', fontsize=16)
plt.legend(prop={'size': 12, 'weight': 'bold'})
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.gca().set_facecolor('lightgrey')
plt.grid(True)
plt.show()

#%%
"""
pH vs CE
"""
#%%
for clasificacion, datos in df.groupby('Textura'):
    plt.scatter(datos['CE'], datos['pH'], color=colores_textura[clasificacion], label=clasificacion)

plt.xlabel('CE', fontsize=14, fontweight='bold')
plt.ylabel('pH', fontsize=14, fontweight='bold')
plt.title('CE vs pH', fontsize=16)
plt.legend(prop={'size': 12, 'weight': 'bold'})
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.gca().set_facecolor('gainsboro')
plt.grid(True)
plt.show()

#%%
"""
%Humedad vs CE
"""
#%%
for clasificacion, datos in df.groupby('Clasificación_Suelo'):
    plt.scatter(datos['CE'], datos['Humedad'], color=colores_suelo[clasificacion], label=clasificacion)

plt.xlabel('CE', fontsize=14, fontweight='bold')
plt.ylabel('Humedad', fontsize=14, fontweight='bold')
plt.title('CE vs Humedad', fontsize=16)
plt.legend(prop={'size': 12, 'weight': 'bold'})
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.gca().set_facecolor('gainsboro')
plt.grid(True)
plt.show()

#%%
"""
Texturas vs CE
"""
#%%
texturas_mapeo = {
    'Franco arenoso': 1,
    'Franco': 2,
    'Franco limoso': 3,
    'Franco arcillo limoso': 4,
    'Franco arcilloso': 5,
    'Arcilloso': 6,
    'Arenoso': 7
}

df['Textura_Num'] = df['Textura'].map(texturas_mapeo)

for clasificacion, datos in df.groupby('Clasificación_CE'):
    plt.scatter(datos['Textura_Num'], datos['CE'], color=colores_ce[clasificacion], label=clasificacion)

plt.xlabel('')
plt.ylabel('CE', weight='bold', size=12)
plt.title('Gráfico de log_CE vs Textura')
plt.legend()
plt.grid(True)

# Establecer las etiquetas del eje X
etiquetas_texto = [k for k, _ in sorted(texturas_mapeo.items(), key=lambda item: item[1])]
plt.xticks(range(1, len(etiquetas_texto) + 1), etiquetas_texto, weight='bold', size=12)
plt.yticks(weight='bold', size=8)

plt.show()


#%%
"""
Cobertura vs CE
"""
#%%
for clasificacion, datos in df.groupby('Clasificación_CE'):
    plt.scatter(datos['CE'], datos['Vegetacion_porc'], color=colores_ce[clasificacion], label=clasificacion)

plt.xlabel('CE', fontsize=14, fontweight='bold')
plt.ylabel('Cobetura vegetal', fontsize=14, fontweight='bold')
plt.title('CE vs Cobertura vegetal', fontsize=16)
plt.legend(prop={'size': 12, 'weight': 'bold'})
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.gca().set_facecolor('gainsboro')
plt.grid(True)
plt.show()

#%%
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(9, 5))
axes = axes.flat
columnas_numericas = df.select_dtypes(include=['float64', 'int']).columns
#columnas_numericas = columnas_numericas.drop('Latitud')
#columnas_numericas = columnas_numericas.drop('Longitud')

for i, colum in enumerate(columnas_numericas):
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
    axes[i].set_title(colum, fontsize = 12, fontweight = "bold")
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel("")
    
    
fig.tight_layout()
plt.subplots_adjust(top = 0.9)
fig.suptitle('Distribución variables numéricas', fontsize = 10, fontweight = "bold");

#%% DATOS SENSOR

datos_sensor = df.drop(columns=[col for col in df.columns if col not in ["Sensor_hum", "Sensor_temp", "Sensor_CE_suelo",
                                                                         "Sensor_CE_agua", "Sensor_perm_r", "Sensor_perm_i",
                                                                         "CE", "Humedad", "RAS", "Clasificación_CE", 'CE_map']])
datos_sensor = datos_sensor.dropna()
datos_sensor["Relacion_i_sobre_r"] = datos_sensor["Sensor_perm_i"] / datos_sensor["Sensor_perm_r"]

sns.set(style="darkgrid")
plt.figure(facecolor='white', figsize=(8, 6), dpi=100)
paleta = sns.color_palette("rainbow", n_colors=5)
sns.scatterplot(data=datos_sensor, x='Sensor_CE_suelo', y='CE', 
                hue='Clasificación_CE', palette=paleta)
plt.plot([0, 100], [0, 100], ls = '--', color='darkred')
plt.xlabel('CE sensor (dS/m)')
plt.ylabel('CE laboratorio (dS/m)')
plt.xlim(0, 5)
plt.ylim(0, 200)
plt.legend(title="Clasificación")
plt.show()

sns.set(style="darkgrid")
plt.figure(facecolor='white', figsize=(16, 10), dpi=120)
paleta = sns.color_palette("rainbow", n_colors=5)
scatter_plot = sns.scatterplot(data=datos_sensor, x='Sensor_hum', y='Humedad', 
                hue='Clasificación_CE', palette=paleta, s=100)  # Aumentar el tamaño de los puntos ploteados
plt.xlabel('Humedad sensor (%)', fontsize=14, fontweight='bold')
plt.ylabel('Humedad laboratorio (%)', fontsize=14, fontweight='bold')
plt.legend(title="Clasificación", fontsize=14, prop={'size': 12, 'weight': 'bold'})
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.plot([0, 100], [0, 100], ls='--', color='darkred')
plt.xlim(0, 50)
plt.ylim(0, 25)
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(datos_sensor["CE"], datos_sensor["Sensor_perm_i"], s=10, c='b')
plt.xlabel('CE laboratorio (dS/m)')
plt.ylabel('\u03B5 imag')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(datos_sensor["Sensor_CE_suelo"], datos_sensor["Sensor_perm_i"], s=10, c='b')
plt.xlabel('CE sensor (dS/m)')
plt.ylabel('\u03B5 imag')
plt.show()

#%%
datos_sensor['Error_CE'] = datos_sensor['CE'] - datos_sensor['Sensor_CE_suelo']

bins=[0, 2, 4, 8, 16, 50, 100, 200]
datos_sensor['bins_error_CE'] = pd.cut(datos_sensor['CE'], bins=bins)


fig, ax1 = plt.subplots(nrows=1, figsize=(8, 12))
sns.boxplot(data=datos_sensor , x='bins_error_CE', y='Error_CE', ax=ax1)
ax1.set_title('Error por intervalos - CE laboratorio - CE sensor')
ax1.set_xlabel('Intervalos')
ax1.set_ylabel('Error CE')

#%% Permitividad real e imaginaria, cruce de curvas
colores_ce_sensor = {
    'No salino': 'blue',
    'Ligeramente salino': 'green',
    'Moderadamente salino': 'yellow',
    'Fuertemente salino': 'orange',
    'Extremadamente salino': 'red'
}

colores_puntos = [colores_ce_sensor[clase] for clase in datos_sensor['Clasificación_CE']]

# Ajuste de regresión polinómica de grado 2 para el eje y principal
coeficientes_perm_r = np.polyfit(datos_sensor['Sensor_hum'], datos_sensor['Sensor_perm_r'], 2)
x_fit_perm_r = np.linspace(min(datos_sensor['Sensor_hum']), max(datos_sensor['Sensor_hum']), 100)
y_fit_perm_r = np.polyval(coeficientes_perm_r, x_fit_perm_r)

# Ajuste de regresión polinómica de grado 2 para el eje y secundario
coeficientes_perm_i = np.polyfit(datos_sensor['Sensor_hum'], datos_sensor['Sensor_perm_i'], 2)
x_fit_perm_i = np.linspace(min(datos_sensor['Sensor_hum']), max(datos_sensor['Sensor_hum']), 100)
y_fit_perm_i = np.polyval(coeficientes_perm_i, x_fit_perm_i)

fig, ax1 = plt.subplots() #gráfico

# Eje y
ax1.scatter(datos_sensor['Sensor_hum'], datos_sensor['Sensor_perm_r'], c=colores_puntos, 
            label='Sensor_perm_r')
ax1.plot(x_fit_perm_r, y_fit_perm_r, color='black')
ax1.set_xlabel('Humedad (%)')
ax1.set_ylabel('Permitividad real', color='black')
ax1.grid(True)

# Segundo eje y
ax2 = ax1.twinx()
ax2.scatter(datos_sensor['Sensor_hum'], datos_sensor['Sensor_perm_i'], c=colores_puntos, 
            marker='x', label='Sensor_perm_i')
ax2.plot(x_fit_perm_i, y_fit_perm_i, color='blue', linestyle='-', label='Ajuste_perm_i')
ax2.set_ylabel('Permitividad imaginaria', color='black')

legend_elements = [
    plt.Line2D([], [], color='none', label='Permitividad sensor', alpha=0),
    plt.Line2D([0], [0], color='blue', label='Permitividad imaginaria'),
    plt.Line2D([0], [0], color='black', label='Permitividad real'),
    plt.Line2D([], [], color='none', label='', alpha=0),
    plt.Line2D([], [], color='none', label='Salinidad', alpha=0),
    plt.Line2D([], [], marker='o', markersize=8, color='red', linestyle='None', label='Extremadamente salino'),
    plt.Line2D([], [], marker='o', markersize=8, color='orange', linestyle='None', label='Fuertemente salino'),
    plt.Line2D([], [], marker='o', markersize=8, color='yellow', linestyle='None', label='Moderadamente salino'),
    plt.Line2D([], [], marker='o', markersize=8, color='green', linestyle='None', label='Ligeramente salino'),
    plt.Line2D([], [], marker='o', markersize=8, color='blue', linestyle='None', label='No salino')
]

# Agregar la leyenda al gráfico
plt.legend(handles=legend_elements, loc='upper left')

# Ajustar tamaño de fuente de los valores de los ejes x e y
ax1.tick_params(axis='both', which='major', labelsize=12)
ax2.tick_params(axis='both', which='major', labelsize=12)

# Ajustar tamaño de fuente de las etiquetas
ax1.set_xlabel('Humedad (%)', fontsize=16)
ax1.set_ylabel('Permitividad real', color='black', fontsize=16)
ax2.set_ylabel('Permitividad imaginaria', color='black', fontsize=16)

plt.show()

#%%
for clasificacion, datos_sensor in df.groupby('Clasificación_CE'):
    plt.scatter(datos_sensor['Sensor_perm_r'], datos_sensor['CE'], color=colores_ce[clasificacion], label=clasificacion)

plt.xlabel('εr', fontsize=20, fontweight='bold')
plt.ylabel('CE', fontsize=18, fontweight='bold')
plt.title('Permitividad real vs CE saturación', fontsize=16)
plt.legend(prop={'size': 15, 'weight': 'bold'})
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.gca().set_facecolor('gainsboro')
plt.grid(True)
plt.show()

for clasificacion, datos_sensor in df.groupby('Clasificación_CE'):
    plt.scatter(datos_sensor['Sensor_perm_i'], datos_sensor['CE'], color=colores_ce[clasificacion], label=clasificacion)

plt.xlabel('εi', fontsize=20, fontweight='bold')
plt.ylabel('CE', fontsize=18, fontweight='bold')
plt.title('Permitividad imaginaria vs CE saturación', fontsize=16)
plt.legend(prop={'size': 15, 'weight': 'bold'})
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.gca().set_facecolor('gainsboro')
plt.grid(True)
plt.show()

#%%
sns.set(style="darkgrid")

# Crear una figura y ejes para los cuatro gráficos simultáneos
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

# Gráfico 1: CE vs Sensor_perm_r con Hue=Clasificación_CE
sns.scatterplot(data=datos_sensor, x='Sensor_perm_r', y='CE', hue='Clasificación_CE', ax=axes[0, 0])
axes[0, 0].set_xlabel("εr", fontsize=18)
axes[0, 0].set_ylabel("CE saturación", fontsize=18)
axes[0, 0].legend(title="Salinidad (saturación)", fontsize=12, loc='lower right')

# Gráfico 2: CE vs Sensor_perm_i con Hue=Clasificación_CE
sns.scatterplot(data=datos_sensor, x='Sensor_perm_i', y='CE', hue='Sensor_CE_suelo', ax=axes[0, 1])
axes[0, 1].set_xlabel("εi", fontsize=18)
axes[0, 1].set_ylabel("CE saturación", fontsize=18)
axes[0, 1].legend(title="Salinidad (sensor)", fontsize=12, loc='lower right')

# Gráfico 3: CE vs Sensor_perm_r con Hue=Sensor_CE_suelo
sns.scatterplot(data=datos_sensor, x='Sensor_perm_r', y='CE', hue='Humedad', ax=axes[1, 0])
axes[1, 0].set_xlabel("εr", fontsize=18)
axes[1, 0].set_ylabel("CE saturación", fontsize=18)
axes[1, 0].legend(title="Humedad", fontsize=12, loc='lower right')

# Gráfico 4: CE vs Sensor_perm_i con Hue=Sensor_CE_suelo
sns.scatterplot(data=datos_sensor, x='Sensor_perm_i', y='CE', hue='Humedad', ax=axes[1, 1])
axes[1, 1].set_xlabel("εi", fontsize=18)
axes[1, 1].set_ylabel("CE saturación", fontsize=18)
axes[1, 1].legend(title="Humedad", fontsize=12, loc='lower right')

# Ajustar el espacio entre los gráficos para que no se superpongan títulos y leyendas
plt.tight_layout()

plt.show()

