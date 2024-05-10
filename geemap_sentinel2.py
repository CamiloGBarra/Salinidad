#%%
import geemap
import ee

ee.Initialize()
#%%
shapefile = r"C:\Users\camil\Downloads\Salinidad\prueba_boruta\Descomposiciones\QGis\area_salinida\borrar.shp"
roi = geemap.shp_to_ee(shapefile)

#%%
coleccion = ee.ImageCollection("COPERNICUS/S2")

fecha_inicio = "2023-01-01"
fecha_fin    = "2023-08-31"

procentaje_nubosidad = 5

#%%
sentinel2 = (coleccion.filterBounds(roi)
             .filterDate(ee.Date(fecha_inicio), ee.Date(fecha_fin))
             .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", procentaje_nubosidad))
            )

numero_imagenes = sentinel2.size().getInfo()
print(f"Se encontraron {numero_imagenes} imágenes")

#%% Información de las imágenes: ID, fecha y nubosidad
for image in sentinel2.toList(sentinel2.size()).getInfo():
    image_id = image['id']
    image_date = ee.Date(image['properties']['system:time_start']).format('YYYY-MM-dd').getInfo()
    cloud_percentage = image['properties']['CLOUDY_PIXEL_PERCENTAGE']
    
    print(f"Image ID: {image_id}")
    print(f"Date: {image_date}")
    print(f"Cloud Percentage: {cloud_percentage}%")
    print("=" * 40)

#%%
imagen = sentinel2.mean()
mosaico = imagen.clip(roi)

#%%
def ndvi(imagen_ndvi):
    ndvi = imagen_ndvi.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return imagen_ndvi.addBands(ndvi)

coleccion_ndvi = sentinel2.map(ndvi)

NDVI = ee.Image(coleccion_ndvi.first())

Map = geemap.Map()
Map.centerObject(roi, zoom=10)
Map.addLayer(NDVI.select("NDVI"), {
    'palette': ['red', 'yellow', 'green']
    }, 'NDVI')
Map.addLayerControl()
Map

html_path = r"C:\Users\camil\Downloads\mapa.html"
Map.to_html(html_path)

#%%
Map = geemap.Map()
Map.centerObject(roi, zoom=10)
Map.addLayer(mosaico, {
    'bands': ['B4', 'B3', 'B2'],
    'min'  : 0,
    'max'  : 3000,
    'gamma': 1
    }, "RGB")

Map.addLayer(roi, {}, "ROI")
Map.addLayerControl()
Map


#%%
html_path = r"C:\Users\camil\Downloads\mapa.html"
Map.to_html(html_path)

#%%
# Definir las bandas a exportar
bands_to_export = ['B4', 'B3', 'B2']  # Cambia según las bandas que necesitas

# Opciones de exportación
export_options = {
    'scale': 0.0001,
    'region': roi,
    'fileFormat': 'GeoTIFF',
}

# Exportar la imagen
task = ee.batch.Export.image.toDrive(imagen.select(bands_to_export), 
                                     description='imagen_sentinel2', 
                                     **export_options)
task.start()

print("Exportando imagen...")




