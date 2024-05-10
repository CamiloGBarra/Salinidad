import geopandas as gpd
import rasterio
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt

# Cargar el shapefile con los puntos
shapefile_path = "C://Users//camil//Downloads//python_salinidad//shapes//puntos.shp"
points = gpd.read_file(shapefile_path)

# Extraer los valores de los rasters en los puntos
raster_path = "C://Users//camil//Downloads//python_salinidad//shapes//optico_s2.tif"
ndvi_path = "C://Users//camil//Downloads//python_salinidad//shapes//ndvi.tif"
ndwi_path = "C://Users//camil//Downloads//python_salinidad//shapes//ndwi.tif"
si1_path = "C://Users//camil//Downloads//python_salinidad//shapes//si1.tif"

def extract_pixel_values(points, raster_path):
    dataset = rasterio.open(raster_path)
    bands = dataset.read()
    num_bands = bands.shape[0]
    
    values = []
    for i in range(num_bands):
        band_values = []
        for point in points.geometry:
            row, col = dataset.index(point.x, point.y)
            pixel_value = bands[i][row, col]
            band_values.append(pixel_value)
        values.append(band_values)
    
    return values

# Extraer los valores de las 16 bandas
band_values = extract_pixel_values(points, raster_path)

# Extraer los valores de los índices espectrales
ndvi_values = extract_pixel_values(points, ndvi_path)
ndwi_values = extract_pixel_values(points, ndwi_path)
si1_values = extract_pixel_values(points, si1_path)

# Crear un DataFrame con las columnas de los valores extraídos
df = pd.DataFrame({
    'CE.2': points['CE.2'],
    'RAS': points['RAS'],
    'ndvi': ndvi_values,
    'ndwi': ndwi_values,
    'si1': si1_values
})

# Agregar la columna de clasificación
def classify_salininity(ce_value):
    if ce_value <= 2:
        return 'No Salino'
    elif ce_value <= 4:
        return 'Ligeramente Salino'
    elif ce_value <= 8:
        return 'Salinidad Media'
    elif ce_value <= 16:
        return 'Fuertemente Salino'
    else:
        return 'Extremadamente Salino'

df['Clasificacion'] = df['CE.2'].map(classify_salininity)

# Mapear las clasificaciones a valores numéricos
classification_mapping = {
    'No Salino': 0,
    'Ligeramente Salino': 1,
    'Salinidad Media': 2,
    'Fuertemente Salino': 3,
    'Extremadamente Salino': 4
}

df['Clasificacion_Num'] = df['Clasificacion'].map(classification_mapping)

# Dividir los datos en conjuntos de entrenamiento y prueba
X = df.drop(['CE.2', 'RAS', 'Clasificacion', 'Clasificacion_Num'], axis=1)
y = df['Clasificacion_Num']

# Entrenar el modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Realizar predicciones en los datos de entrenamiento
y_pred = model.predict(X)

# Calcular la matriz de confusión y el índice Kappa
confusion_mtx = confusion_matrix(y, y_pred)
kappa_score = cohen_kappa_score(y, y_pred)

# Visualizar la matriz de confusión
labels = list(classification_mapping.keys())
plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión')
plt.colorbar()
plt.xticks(np.arange(len(labels)), labels, rotation=45)
plt.yticks(np.arange(len(labels)), labels)
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Verdadera')
plt.show()

print("Índice Kappa:", kappa_score)

# Visualizar el mapa clasificado de salinidad
def classify_raster(pixel_values):
    X = np.array(pixel_values).T
    salinity_pred = model.predict(X)
    return salinity_pred.reshape(dataset.height, dataset.width)

# Cargar el raster para obtener los metadatos
with rasterio.open(raster_path) as dataset:
    salinity_map = classify_raster(band_values)

plt.imshow(salinity_map, cmap='RdYlGn', vmin=0, vmax=4)
plt.title('Mapa Clasificado de Salinidad')
plt.colorbar(label='Clasificación')
plt.show()
