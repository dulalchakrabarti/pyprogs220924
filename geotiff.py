from osgeo import gdal
from osgeo import gdalconst
import struct
fl = open('ndvi_24-08_05-06_2024.csv','w+')
fl.write('lat'+','+'lon'+','+'ndvi'+'\n')
# Open tif file
ds = gdal.Open('ndvi.tif')
# GDAL affine transform parameters, According to gdal documentation xoff/yoff are image left corner, a/e are pixel width/height and b/d is rotation and is zero if image is north up. 
xoff, a, b, yoff, d, e = ds.GetGeoTransform()
def pixel2coord(x, y):
    """Returns global coordinates from pixel x, y coords"""
    xp = a * x + b * y + xoff
    yp = d * x + e * y + yoff
    return(xp, yp)
colms = ds.RasterXSize
rows = ds.RasterYSize
bands = ds.RasterCount
band = ds.GetRasterBand(1)
bandtype = gdal.GetDataTypeName(band.DataType)
count = 0
for row in range(rows):
 for col in range(colms):
  lon, lat = pixel2coord(col, row)
  pix = band.ReadRaster( col, row, 1,1,1, 1, band.DataType)
  val = struct.unpack('b', pix)
  count = count +1
  lat = round(lat,1)
  lon = round(lon,1)
  if lat > 5 and lat < 40 and lon >60 and lon < 100 :
   print(lat, lon, val[0])
   fl.write(str(lat)+','+str(lon)+','+str(val[0])+'\n')
print(count)