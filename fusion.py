#%%
import rasterio
import os
import glob
import numpy as np
from geoarray import GeoArray
from arosics import COREG

#%%
im_reference = r'C:\Users\camil\Downloads\fusion\feature_space_resultados\S1A_OPER_SAR_EOSSP__CORE_L1A_OLF_20230418T122622_Cal_mat_Spk_TC_Yamaguchi.tif'
im_target    = r"C:\Users\camil\Downloads\fusion\feature_space_resultados\SAI9.tif"

#%%
geoArr  = GeoArray(im_reference)

#%%
ref_ndarray = geoArr[:]            # numpy.ndarray with shape (10980, 10980)
ref_gt      = geoArr.geotransform  # GDAL geotransform: (300000.0, 10.0, 0.0, 5900040.0, 0.0, -10.0)
ref_prj     = geoArr.projection

#%%
# get a sample numpy array with corresponding geoinformation as target image
geoArr  = GeoArray(im_target)

tgt_ndarray = geoArr[:]            # numpy.ndarray with shape (10980, 10980)
tgt_gt      = geoArr.geotransform  # GDAL geotransform: (300000.0, 10.0, 0.0, 5900040.0, 0.0, -10.0)
tgt_prj     = geoArr.projection    # projection as WKT string ('PROJCS["WGS 84 / UTM zone 33N....')

# create in-memory instances of GeoArray from the numpy array data, the GDAL geotransform tuple and the WKT
# projection string
geoArr_reference = GeoArray(ref_ndarray, ref_gt, ref_prj)
geoArr_target    = GeoArray(tgt_ndarray, tgt_gt, tgt_prj)

#%%
CR = COREG(geoArr_reference, geoArr_target, wp=(354223, 5805559), ws=(256,256))
CR.calculate_spatial_shifts()

#%%
CR.correct_shifts()



