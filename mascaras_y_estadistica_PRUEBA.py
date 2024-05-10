#%%
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import Point
import numpy as np

#%%
def extract_raster_values(raster_path, polygons_path, points_path):
    # Leer el raster
    with rasterio.open(raster_path) as src:
        raster_data = src.read(1)  # asumimos que estamos trabajando con una banda

    # Leer los polígonos y los puntos usando geopandas
    polygons = gpd.read_file(polygons_path)
    points = gpd.read_file(points_path)

    # Aplicar la máscara de los polígonos al raster
    mask = geometry_mask(polygons.geometry, out_shape=raster_data.shape, transform=src.transform, invert=True)
    raster_data_masked = np.ma.masked_array(raster_data, mask)

    # Extraer valores del raster en ubicaciones de puntos
    values_at_points = []
    for point in points.geometry:
        # Verificar si el punto está dentro de la máscara del polígono
        if not mask[int(point.y), int(point.x)]:
            # Si no está en la máscara, extraer el valor del raster en el punto
            value = raster_data[int(point.y), int(point.x)]
            values_at_points.append(value)
        else:
            values_at_points.append(None)  # Opcional: se puede manejar puntos fuera de la máscara de alguna manera específica

    return values_at_points

#%%
raster_path = r"C:\Users\camil\Downloads\Salinidad\imagenes SALINIDAD\procesadas\PP\EOL1ASARSAO1A6877636_2023_04_06_orbit_DESCENDING\con speckle_refinedlee\ALP.tif"
polygons_path = r"C:\Users\camil\Downloads\Salinidad\Suelo desnudo\suelo_desnudo_ndvi_menos0_25_shape.shp"
points_path = r"C:\Users\camil\Downloads\Salinidad\Primera campaña SALINIDAD\Campaña\relevamiento_abril23\relevamiento_abril23.shp"

#%%
result = extract_raster_values(raster_path, polygons_path, points_path)

#%%
print(result)
