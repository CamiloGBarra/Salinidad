#%%
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf

#%%
Sentinel2_mediaAnual_2022abr_2023abr = r'C:\Users\camil\Downloads\Salinidad\imagenes SALINIDAD\Sentinel-2_RGB\anual_abr_2022_2023.tif'
with rasterio.open(Sentinel2_mediaAnual_2022abr_2023abr) as src:
    transformacion = src.transform
    bandas = src.read()

#%% Sentinel2_Anual_2022abr_2023abr
ruta_puntos = 'C://Users//camil//Downloads//Salinidad//python_salinidad//shapes//puntos_abr2023.shp'
puntos = gpd.read_file(ruta_puntos)

valores_indices = []
for punto in puntos.geometry:
    coordenada = (punto.x, punto.y)
    fila, columna = rasterio.transform.rowcol(transformacion, coordenada[0], coordenada[1])
    ndvi = (bandas[8, fila, columna] - bandas[4, fila, columna]) / (bandas[8, fila, columna] + bandas[4, fila, columna])
    ndsi = (bandas[4, fila, columna] - bandas[8, fila, columna]) / (bandas[4, fila, columna] + bandas[8, fila, columna])
    ndwi = (bandas[3, fila, columna] - bandas[11, fila, columna]) / (bandas[3, fila, columna] + bandas[11, fila, columna])
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
    valores_indices.append([ndvi, ndsi, ndwi, evi, savi, osavi, crsi, khaier, gari, gdvi2, bi, ndi, tbi, si1, si2, si3,
                            si4, sai1, sai2, sai3, sai4, sai5, sai6, sai7, sai8, sai9, corsi, vssi, bsi])

# convertir los valores de los índices en un array numpy
valores_indices = np.array(valores_indices)

#%%
ce = puntos['CE']
df_indices = pd.DataFrame({'CE': ce})
df_indices['NDVI'] = valores_indices[:, 0]
df_indices['NDSI'] = valores_indices[:, 1]
df_indices['NDWI'] = valores_indices[:, 2]
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

#%%  ANTES
columnas = df_indices.columns.difference(['CE'])
fig, ax = plt.subplots()
df_indices[columnas].boxplot(ax=ax)
ax.set_xticklabels(columnas, rotation=90, ha='right')
plt.xlabel('Índices')
plt.ylabel('Valores')
plt.show()

#%%
scaler = StandardScaler()
columnas = df_indices.columns
df_indices[columnas] = scaler.fit_transform(df_indices[columnas])

#%%  DESPUÉS
columnas = df_indices.columns.difference(['CE'])
fig, ax = plt.subplots()
df_indices[columnas].boxplot(ax=ax)
ax.set_xticklabels(columnas, rotation=90, ha='right')
plt.xlabel('Índices')
plt.ylabel('Valores')
plt.show()

#%%
fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(df_indices["NDVI"], df_indices["BI"], estilo, color=color)
plt.xlabel("NDVI")
plt.ylabel("BI")
plt.show()

fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(df_indices["SI1"], df_indices["BI"], estilo, color=color)
plt.xlabel("SI1")
plt.ylabel("BI")
plt.show()

fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(df_indices["NDVI"], df_indices["SI1"], estilo, color=color)
plt.xlabel("NDVI")
plt.ylabel("SI1")
plt.show()

fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(df_indices["SAI8"], df_indices["NDWI"], estilo, color=color)
plt.xlabel("SAI8")
plt.ylabel("NDWI")
plt.show()

#%%
NDVI_BI = pd.DataFrame({'NDVI_BI': (((1 - df_indices['BI'])**2) + df_indices['NDVI'])**(1/2),
                         'CE': df_indices['CE']}).dropna()

SI1_BI = pd.DataFrame({'SI1_BI': ((df_indices["SI1"]**2)+df_indices["BI"])**(1/2),
                         'CE': df_indices['CE']}).dropna()

NDVI_SI1 = pd.DataFrame({'NDVI_SI1': (((df_indices["NDVI"]-1)**2)+(df_indices["SI1"]**2))**(1/2),
                         'CE': df_indices['CE']}).dropna()

SAI8_NDWI = pd.DataFrame({'SAI8_NDWI': (((df_indices["SAI8"]-1)**2)+(df_indices["NDWI"]**2))**(1/2),
                         'CE': df_indices['CE']}).dropna()

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

######################  SAI8-NDWI ####################################
corr_pearson = SAI8_NDWI["SAI8_NDWI"].corr(SAI8_NDWI["CE"], method="pearson")
corr_spearman = SAI8_NDWI["SAI8_NDWI"].corr(SAI8_NDWI["CE"], method="spearman")
corr_pearson_1 = pearsonr(x = SAI8_NDWI["SAI8_NDWI"], y =  SAI8_NDWI["CE"])
print("P-value: ", corr_pearson_1[1])

print(f"Correlación de Pearson SAI8_NDWI-CE: {corr_pearson}")
print(f"Correlación de Spearman SAI8_NDWI-CE: {corr_spearman}")

fig, ax = plt.subplots()
estilo= "o"
color = "steelblue"
ax.plot(SAI8_NDWI["CE"], SAI8_NDWI["SAI8_NDWI"], estilo, color=color)
plt.xlabel("CE")
plt.ylabel("SAI8_NDWI")
plt.show()

#%%
"""
REGRESIÓN LINEAL SIMPLE
"""
#%%  AJUSTE DEL MODELO
# División de los datos en train y test
# ==============================================================================
X = pd.DataFrame(SI1_BI["SI1_BI"])
y = pd.DataFrame(SI1_BI["CE"])

X_train, X_test, y_train, y_test = train_test_split(
                                        X.values.reshape(-1,1),
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )

# Creación del modelo
# ==============================================================================
modelo_SI1_BI = LinearRegression()
modelo_SI1_BI.fit(X = X_train.reshape(-1, 1), y = y_train)
#%%
# Información del modelo
# ==============================================================================
print("Intercept:", modelo_SI1_BI.intercept_)
print("Coeficiente:", list(zip(X.columns, modelo_SI1_BI.coef_.flatten(), )))
print("Coeficiente de determinación R^2:", modelo_SI1_BI.score(X, y))

#%%
# Error de test del modelo 
# ==============================================================================
predicciones = modelo_SI1_BI.predict(X = X_test)
print(predicciones[0:3,])

rmse = mean_squared_error(
        y_true  = y_test,
        y_pred  = predicciones,
        squared = False
       )
print("")
print(f"El error (rmse) de test es: {rmse}")

#%%
"""
Statsmodels
La implementación de regresión lineal de Statsmodels, es más completa que la de 
Scikitlearn ya que, además de ajustar el modelo, permite calcular los test estadísticos 
y análisis necesarios para verificar que se cumplen las condiciones sobre las que 
se basa este tipo de modelos. Statsmodels tiene dos formas de entrenar el modelo:

Indicando la fórmula del modelo y pasando los datos de entrenamiento como un dataframe 
que incluye la variable respuesta y los predictores. Esta forma es similar a la utilizada 
en R.

Pasar dos matrices, una con los predictores y otra con la variable respuesta. 
Esta es igual a la empleada por Scikitlearn con la diferencia de que a la matriz 
de predictores hay que añadirle una primera columna de 1s.
"""
#%%
# División de los datos en train y test
# ==============================================================================
X = pd.DataFrame(SI1_BI["SI1_BI"])
y = pd.DataFrame(SI1_BI["CE"])

X_train, X_test, y_train, y_test = train_test_split(
                                        X.values.reshape(-1,1),
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )

# Creación del modelo
# ==============================================================================
modelo_SI1_BI = LinearRegression()
modelo_SI1_BI.fit(X = X_train.reshape(-1, 1), y = y_train)

# Creación del modelo utilizando el modo fórmula (similar a R)
# ==============================================================================
# datos_train = pd.DataFrame(np.hstack((X_train, y_train)), columns=['bateos', 'runs'])
# modelo = smf.ols(formula = 'runs ~bateos', data = datos_train)
# modelo = modelo.fit()
# print(modelo.summary())
# Creación del modelo utilizando matrices como en scikitlearn
# ==============================================================================
# A la matriz de predictores se le tiene que añadir una columna de 1s para el intercept del modelo
X_train = sm.add_constant(X_train, prepend=True)
modelo_SI1_BI = sm.OLS(endog=y_train, exog=X_train,)
modelo_SI1_BI = modelo_SI1_BI.fit()
print(modelo_SI1_BI.summary())

#%% Intervalos de confianza para los coeficientes del modelo

modelo_SI1_BI.conf_int(alpha=0.05)

#%% Predicciones
# Predicciones con intervalo de confianza del 95%
# ==============================================================================
predicciones = modelo_SI1_BI.get_prediction(exog = X_train).summary_frame(alpha=0.05)
predicciones.head(4)

#%% Representación gráfica del modelo
# Predicciones con intervalo de confianza del 95%
# ==============================================================================
predicciones = modelo_SI1_BI.get_prediction(exog = X_train).summary_frame(alpha=0.05)
predicciones['x'] = X_train[:, 1]
predicciones['y'] = y_train
predicciones = predicciones.sort_values('x')

# Gráfico del modelo
# ==============================================================================
fig, ax = plt.subplots(figsize=(6, 3.84))

ax.scatter(predicciones['x'], predicciones['y'], marker='o', color = "gray")
ax.plot(predicciones['x'], predicciones["mean"], linestyle='-', label="OLS")
ax.plot(predicciones['x'], predicciones["mean_ci_lower"], linestyle='--', color='red', label="95% CI")
ax.plot(predicciones['x'], predicciones["mean_ci_upper"], linestyle='--', color='red')
ax.fill_between(predicciones['x'], predicciones["mean_ci_lower"], predicciones["mean_ci_upper"], alpha=0.1)
ax.legend();

#%%
# Error de test del modelo 
# ==============================================================================
X_test = sm.add_constant(X_test, prepend=True)
predicciones = modelo_SI1_BI.predict(exog = X_test)
rmse = mean_squared_error(
        y_true  = y_test,
        y_pred  = predicciones,
        squared = False
       )
print("")
print(f"El error (RMSE) de test es: {rmse}")