#%%
import os
import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np

#%%
# Rutas a los archivos
carpeta_tif = r'C:/Users/camil/Downloads/MAIE/Introducción a la Teledetección/Trabajo final/QGis/sentinel2_copo/mosaico_sentinel2_2023.tif'
ruta_shapefile = r'C:\Users\camil\Downloads\MAIE\Introducción a la Teledetección\Trabajo final\QGis\Shapes\poligonos\vegetacion.shp'

#%%
def obtener_rutas_tif(carpeta):
    rutas_tif = []
    for ruta_raiz, carpetas, archivos in os.walk(carpeta):
        for archivo in archivos:
            if archivo.lower().endswith('.tif'):
                ruta_completa = os.path.join(ruta_raiz, archivo)
                rutas_tif.append(ruta_completa)
    return rutas_tif

poligonos = gpd.read_file(ruta_shapefile)

# diccionario
estadisticas_poligonos = {}

for ruta_tif in obtener_rutas_tif(carpeta_tif):
    with rasterio.open(ruta_tif) as dataset:
        imagen_nombre = os.path.splitext(os.path.basename(ruta_tif))[0]
        for index, poligono in poligonos.iterrows():
            poligono_id = poligono['id']
            if poligono_id not in estadisticas_poligonos:
                estadisticas_poligonos[poligono_id] = {'id': poligono_id}
            window = dataset.window(*poligono.geometry.bounds)
            data = dataset.read(1, window=window, masked=True)
            if not data.mask.all():
                estadisticas_poligonos[poligono_id][f"{imagen_nombre}_media"] = float(data.mean())
                #estadisticas_poligonos[poligono_id][f"{imagen_nombre}_min"] = float(data.min())
                #estadisticas_poligonos[poligono_id][f"{imagen_nombre}_max"] = float(data.max())
                #estadisticas_poligonos[poligono_id][f"{imagen_nombre}_mediana"] = float(np.median(data))
                #estadisticas_poligonos[poligono_id][f"{imagen_nombre}_std"] = float(data.std())

df_estadisticas = pd.DataFrame(estadisticas_poligonos.values()).set_index('id')

print(df_estadisticas)


#%%
# Separar las medias
media = [col for col in df_estadisticas.columns if col.endswith('_media')]
df_media = df_estadisticas[media]

# Modificar los nombres
nuevos_nombres = [col.replace('_media', '') for col in df_media.columns]
df_media.columns = nuevos_nombres

#%%
# Exportar el DataFrame df_media a un archivo CSV
df_media.to_csv(r"C:\Users\camil\Downloads\vegetacion_sentinel2_2023.csv", index=True)

print(f"DataFrame exportado como '{'indices_polarimetricos_buffer_medias.csv'}'")
