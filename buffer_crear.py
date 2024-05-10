#%%
import geopandas as gpd
from shapely.geometry import Point, Polygon

#%%
# Cargar el shapefile en un GeoDataFrame
ruta_shapefile = r'C:\Users\camil\Downloads\Salinidad\python_salinidad\shapes\cobertura_de_suelo\cobertura_de_suelo1.shp'
gdf = gpd.read_file(ruta_shapefile)

#%%
# Elegir si buffers circulares o cuadrados (True para circulares, False para cuadrados)
usar_buffers_circulares = True
#%%

# Definir el tamaño del buffer en unidades del sistema de coordenadas del GeoDataFrame
tamanio_buffer = 0.00035


#%%
# Crear una nueva columna para almacenar las geometrías de los buffers
gdf['buffer_geometry'] = gdf['geometry'].apply(lambda geom: geom.buffer(tamanio_buffer))

# Función para convertir buffers circulares en cuadrados
def circular_to_square(geom):
    if isinstance(geom, Polygon):
        minx, miny, maxx, maxy = geom.bounds
        center = geom.centroid
        half_side = max(maxx - minx, maxy - miny) / 2
        square_coords = [
            (center.x - half_side, center.y - half_side),
            (center.x + half_side, center.y - half_side),
            (center.x + half_side, center.y + half_side),
            (center.x - half_side, center.y + half_side),
            (center.x - half_side, center.y - half_side)
        ]
        return Polygon(square_coords)
    return geom

# Aplicar la función de conversión si se eligen buffers cuadrados
if not usar_buffers_circulares:
    gdf['buffer_geometry'] = gdf['buffer_geometry'].apply(circular_to_square)

# Crear un nuevo GeoDataFrame con la columna de buffers modificados
columna_geom = 'buffer_geometry' if usar_buffers_circulares else 'geometry'
gdf_buffers = gpd.GeoDataFrame(gdf[columna_geom], geometry=columna_geom, crs=gdf.crs)

#%%
# Guardar el resultado en un nuevo shapefile
tipo_buffer = 'circular' if usar_buffers_circulares else 'cuadrado'
ruta_resultado = fr'C:\Users\camil\Downloads\buffer_modificado_{tipo_buffer}.shp'
gdf_buffers.to_file(ruta_resultado)
