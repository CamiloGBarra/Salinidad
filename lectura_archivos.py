import fiona
import shapely.geometry as sgeom
import pandas as pd
import geopandas as gpd
import re
from rasterstats import zonal_stats

for imag_radar in lista_par_cor:
    obtenerEstadisticas(shpfile, imag_radar, archivo_salida_fuga_buffer, tipo_salida='shp', nodata=-999)
    shpfile = archivo_salida_fuga_buffer

def obtenerEstadisticas(shpfile, raster_entrada, archivo_salida, tipo_salida='shp', nodata=-999):
    # Defino path del shapefile, lo abro con librería fiona y guardo en geometries los shapes correspondientes
    with fiona.open(shpfile) as records:
        geometries = [sgeom.shape(shp['geometry'])
                      for shp in records]
   
    p = re.compile(r'[A-Z]{2,5}')
    s = raster_entrada.split('\\')[-1][20:29]  
    prefix = p.search(s).group()
    print(prefix)
   
    # Calculo las estadísticas de valores mínimos, máximos y promedios
    zs = zonal_stats(geometries, raster_entrada, nodata=nodata, stats=['min', 'max', 'median', 'mean', 'std', 'count'], all_touched=True, prefix = prefix)
    # Abro el shapefile usando la librería GeoPandas
    tabla_shape = gpd.read_file(shpfile)
    # Creo un Dataframe con los valores estadísticos obtenidos
    stats_df = pd.DataFrame(zs)
    # Concateno las dos tablas
    tabla_shape = pd.concat([tabla_shape, stats_df], axis=1)
    # Escribo los resultados al disco
    if tipo_salida == 'shp':
        tabla_shape.to_file(archivo_salida)
    if tipo_salida == 'csv':
        tabla_shape.drop('geometry',axis=1).to_csv(archivo_salida)

    return
