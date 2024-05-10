#%%
import os
import glob
import rasterio

#%%
# ruta del archivo raster
file_path = r"C:\Users\camil\Downloads\fusion\descomposiciones_alineados\PAULI.tif"

# carpeta de salida
output_folder = os.path.dirname(file_path)

# abrir el archivo raster
dataset = rasterio.open(file_path)

# bandas
num_bands = dataset.count

for band in range(1, num_bands + 1):
    band_data = dataset.read(band)
    band_meta = dataset.meta.copy()
    band_meta.update({
        'count': 1,  # Establecer el número de bandas a 1
        'dtype': band_data.dtype  # Establecer el tipo de datos de la banda
    })

    output_filename = f"Pauli{band}.tif"  # Nombre de archivo de salida
    output_path = os.path.join(output_folder, output_filename)  # Ruta de salida completa
    with rasterio.open(output_path, 'w', **band_meta) as dst:
        dst.write(band_data, 1)  # Escribir la banda en el nuevo archivo

# Cerrar el dataset
dataset.close()


#%% NO DATA VALUE
folder_path = r"C:\Users\camil\Downloads\fusion\descomposiciones_alineados"
band_pattern = "banda*.tif"
band_files = glob.glob(os.path.join(folder_path, band_pattern))

nodata_values = [0, -9999]

# Iterar sobrelos archivos de las bandas
for band_file in band_files:
    dataset = rasterio.open(band_file, 'r+')

    for nodata_value in nodata_values:
        dataset.nodata = nodata_value
        band_data = dataset.read(1)
        band_data[band_data == nodata_value] = dataset.nodata
        dataset.write_band(1, band_data)

    dataset.close()