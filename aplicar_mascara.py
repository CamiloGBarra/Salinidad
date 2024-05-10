#%%
import geopandas as gpd
import rasterio
import numpy as np
import os
import rasterio.mask as mask
from os.path import join

#%% MÁSCARA
carpeta_imagenes = r"C:/Users/camil/Downloads/PRUEBA PARA SUELO DESNUDO/imagenes de input"
carpeta_salida = r"C:\Users\camil\Downloads\PRUEBA PARA SUELO DESNUDO"
gdf = gpd.read_file(r"C:\Users\camil\Downloads\Salinidad\Suelo desnudo\suelo_desnudo_ndvi_menos0_25_shape_recortado.shp")

def mascara(imagen_raster, salida_raster):
    with rasterio.open(imagen_raster) as src:
        out_image, out_transform = mask.mask(src, gdf.geometry, crop=True)


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

#%%
imagenes_en_carpeta = os.listdir(carpeta_imagenes)
for imagen in imagenes_en_carpeta:
    if imagen.endswith('.tif'):
        ruta_imagen = os.path.join(carpeta_imagenes, imagen)
        nombre_salida = os.path.splitext(imagen)[0] + '.tif'
        ruta_salida = os.path.join(carpeta_salida, nombre_salida)
        mascara(ruta_imagen, ruta_salida)
        
#%% INTERSECCIÓN
shapefile_path = r"C:\Users\camil\Downloads\MAIE\Introducción a la Teledetección\Trabajo final\QGis\Shapes\copo.shp"

raster_dir = r'C:\Users\camil\Downloads\MAIE\Introducción a la Teledetección\Trabajo final\QGis\MapBiomas Chaco\imágenes'
output_folder = r"C:\Users\camil\Downloads\MAIE\Introducción a la Teledetección\Trabajo final\QGis\MapBiomas Chaco\imagenes_recortadas"

gdf = gpd.read_file(shapefile_path)

for tif_file in os.listdir(raster_dir):
    if tif_file.endswith(".tif"):
        tif_path = join(raster_dir, tif_file)

        with rasterio.open(tif_path) as src:
            out_image, out_transform = mask.mask(src, gdf.geometry, crop=True)

            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

            output_tif_path = join(output_folder, "" + tif_file)

            with rasterio.open(output_tif_path, "w", **out_meta) as dest:
                dest.write(out_image)

        print(f"Recortado y guardado: {output_tif_path}")