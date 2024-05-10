#%%
import os
import rasterio
import numpy as np
from osgeo import gdal
from osgeo import ogr
import geopandas as gpd
from rasterio.mask import mask

#%%
# Archivo de entrada y archivo de salida para ALP
input_file_alp = r"C:\Users\camil\Downloads\Salinidad\imagenes SALINIDAD\procesadas\PP\EOL1ASARSAO1A6877636_2023_04_06_orbit_DESCENDING\con speckle_refinedlee\ALP.tif"
output_file_alp = r"C:\Users\camil\Downloads\Salinidad\Región A del ploteo H-Alpha\ALP_recortado.tif"

# Archivo de entrada y archivo de salida para ENT
input_file_ent = r"C:\Users\camil\Downloads\Salinidad\imagenes SALINIDAD\procesadas\PP\EOL1ASARSAO1A6877636_2023_04_06_orbit_DESCENDING\con speckle_refinedlee\ENT.tif"
output_file_ent = r"C:\Users\camil\Downloads\Salinidad\Región A del ploteo H-Alpha\ENT_recortado.tif"

#%%
umbral_alp = 42
umbral_ent = 0.5

#%%
with rasterio.open(input_file_alp) as src:
    data_alp = src.read(1)
    data_binaria_alp = np.where(data_alp <= umbral_alp, 1, 0)
    profile_alp = src.profile
    profile_alp.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(output_file_alp, 'w', **profile_alp) as dst:
        dst.write(data_binaria_alp, 1)

print(f"El archivo binario de ALP se ha creado en: {output_file_alp}")

with rasterio.open(input_file_ent) as src:
    data_ent = src.read(1)
    data_binaria_ent = np.where(data_ent <= umbral_ent, 1, 0)
    profile_ent = src.profile
    profile_ent.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(output_file_ent, 'w', **profile_ent) as dst:
        dst.write(data_binaria_ent, 1)

print(f"El archivo binario de ENT se ha creado en: {output_file_ent}")

#%% Vectorización
# Rutas a las imágenes raster binarias
input_raster_path1 = r"C:\Users\camil\Downloads\Salinidad\Región A del ploteo H-Alpha\ALP_recortado.tif"
input_raster_path2 = r"C:\Users\camil\Downloads\Salinidad\Región A del ploteo H-Alpha\ENT_recortado.tif"

# Rutas para guardar los archivos Shapefile resultantes
output_shapefile_path1 = r"C:\Users\camil\Downloads\Salinidad\Región A del ploteo H-Alpha\ALP_vectorizado.shp"
output_shapefile_path2 = r"C:\Users\camil\Downloads\Salinidad\Región A del ploteo H-Alpha\ENT_vectorizado.shp"

# Abre la primera imagen raster
ds1 = gdal.Open(input_raster_path1)

# Comprueba si la apertura del primer raster fue exitosa
if ds1 is None:
    print("No se pudo abrir la primera imagen raster.")
else:
    band1 = ds1.GetRasterBand(1)

    # Crea un nuevo archivo Shapefile para la primera imagen
    driver = ogr.GetDriverByName("ESRI Shapefile")
    output_ds1 = driver.CreateDataSource(output_shapefile_path1)
    output_layer1 = output_ds1.CreateLayer("Poligonos1", srs=None)

    field_defn = ogr.FieldDefn("Valor", ogr.OFTInteger)
    output_layer1.CreateField(field_defn)

    gdal.Polygonize(band1, None, output_layer1, 0, [], callback=None)

    ds1 = None
    output_ds1 = None

    print(f"Vectorización de la primera imagen completada. Los polígonos se han guardado en: {output_shapefile_path1}")

# segunda imagen
ds2 = gdal.Open(input_raster_path2)

if ds2 is None:
    print("No se pudo abrir la segunda imagen raster.")
else:
    # Obtiene la banda de la segunda imagen
    band2 = ds2.GetRasterBand(1)

    # Crea un nuevo archivo Shapefile para la segunda imagen
    output_ds2 = driver.CreateDataSource(output_shapefile_path2)
    output_layer2 = output_ds2.CreateLayer("Poligonos2", srs=None)

    # Define un nuevo campo en la capa para almacenar valores
    field_defn = ogr.FieldDefn("Valor", ogr.OFTInteger)
    output_layer2.CreateField(field_defn)

    # Realiza la vectorización de la segunda imagen
    gdal.Polygonize(band2, None, output_layer2, 0, [], callback=None)

    # Cierra el archivo de la segunda imagen
    ds2 = None
    output_ds2 = None

    print(f"Vectorización de la segunda imagen completada. Los polígonos se han guardado en: {output_shapefile_path2}")
    
#%% Creación de la máscara

gdf1 = gpd.read_file(r"C:\Users\camil\Downloads\Salinidad\Región A del ploteo H-Alpha\ALP_vectorizado.shp")
gdf2 = gpd.read_file(r"C:\Users\camil\Downloads\Salinidad\Región A del ploteo H-Alpha\ENT_vectorizado.shp")

# Filtrar los polígonos que tienen un valor de 1 en ambos dataframes
gdf1 = gdf1[gdf1['Valor'] == 1]
gdf2 = gdf2[gdf2['Valor'] == 1]

# Guardar los resultados en nuevos archivos shapefiles si es necesario
gdf1.to_file(r"C:\Users\camil\Downloads\Salinidad\Región A del ploteo H-Alpha\ALP_mascara.shp")
gdf2.to_file(r"C:\Users\camil\Downloads\Salinidad\Región A del ploteo H-Alpha\ENT_mascara.shp")

### Unión de ambas

# Cargar los dos shapefiles en geodataframes
gdf1 = gpd.read_file(r"C:\Users\camil\Downloads\Salinidad\Región A del ploteo H-Alpha\ALP_mascara.shp")
gdf2 = gpd.read_file(r"C:\Users\camil\Downloads\Salinidad\Región A del ploteo H-Alpha\ENT_mascara.shp")

# Calcular la intersección de los dos geodataframes
intersection = gpd.overlay(gdf1, gdf2, how='intersection')

# Guardar el resultado en un nuevo archivo shape
intersection.to_file(r"C:\Users\camil\Downloads\Salinidad\Región A del ploteo H-Alpha\mascara.shp")

#%%
carpeta_imagenes = r'C:\Users\camil\Downloads\Salinidad\imagenes SALINIDAD\procesadas\PP\EOL1ASARSAO1A6877636_2023_04_06_orbit_DESCENDING\con speckle_refinedlee'
carpeta_salida = r"C:\Users\camil\Downloads\Salinidad\Región A del ploteo H-Alpha\SAOCOM"
gdf = gpd.read_file(r"C:\Users\camil\Downloads\Salinidad\Región A del ploteo H-Alpha\mascara.shp")

def recortar_y_agregar_nodata(imagen_raster, salida_raster):
    with rasterio.open(imagen_raster) as src:
        out_image, out_transform = mask(src, gdf.geometry, crop=True)

        out_image = np.where(out_image == 0, 0, out_image)

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "nodata": 0
        })

        with rasterio.open(salida_raster, 'w', **out_meta) as dest:
            dest.write(out_image)

imagenes_en_carpeta = os.listdir(carpeta_imagenes)

for imagen in imagenes_en_carpeta:
    if imagen.endswith('.tif'):
        ruta_imagen = os.path.join(carpeta_imagenes, imagen)
        nombre_salida = os.path.splitext(imagen)[0] + '_mascara.tif'
        ruta_salida = os.path.join(carpeta_salida, nombre_salida)
        recortar_y_agregar_nodata(ruta_imagen, ruta_salida)
























