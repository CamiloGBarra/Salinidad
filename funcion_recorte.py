#%%
import rasterio as rio
import rasterio.mask
import fiona

#%%

def cortar_lista_imagenes(lista_tiff_a_cortar, shapefile_a_usar):
    '''
    Realiza el corte de la lista de imágenes tiff pasada en el argumento, usando el archivo vectorial en formato .shp

    Argumentos:
    lista_tiff_a_cortar (list): Lista de la ruta de los archivos tiff a los que realiza el corte.
    shapefile_a_usar (str): Ruta del archivo vectorial a usar para el corte.
    '''
    with fiona.open(shapefile_a_usar, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    for imagen in lista_tiff_a_cortar:
        with rio.open(imagen) as src:
            out_image, out_transform = mask.mask(src, shapes, crop=True)
            out_meta = src.meta

        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

        with rio.open(imagen[:-4]+'_recorte.tif', "w", **out_meta) as dest:
            dest.write(out_image)