import imdlib as imd
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
start_yr = 1901
end_yr = 2000
variable = 'rain' # other options are ('tmin'/ 'tmax')

imd.get_data(variable, start_yr, end_yr, fn_format='yearwise', file_dir='./')
data = imd.open_data(variable, start_yr, end_yr,'yearwise', './')
ds = data.get_xarray()
print(ds)
