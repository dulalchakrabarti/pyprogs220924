import pandas as pd
from geopy.geocoders import Nominatim
import time
import tabula
import math
count=0
# convert PDF into CSV file
#tabula.convert_into("rf.pdf", "rf.csv", output_format="csv", pages='all')
gl = open('dist_lat_lon.csv','w')
lines = [line.strip('\n') for line in open('rf.csv')]
for line in lines:
 line = line.split(',')
 stn = line[1]
 per = line[-2][:-1]
 cat = line[-1]
 geolocator = Nominatim(user_agent="my_prog", timeout = None)
 location = geolocator.geocode(stn)
 if location != None:
  print(stn,location.latitude, location.longitude,per,cat)
  gl.write(stn+','+str(location.latitude)+','+str(location.longitude)+','+per+','+cat+'\n')
  count+=1
  time.sleep(1.0)
print(count)

