import pandas as pd
from geopy.geocoders import Nominatim
import time
import tabula
import math
dist = {}
lins = [lns.rstrip('\n') for lns in open('dist.csv')]
for lin in lins:
 txt = lin.split(',')
 dist[txt[0]] = [txt[1],txt[2]]
#print(len(dist))
lines =  [line.rstrip('\n') for line in open('dist_lat_lon.csv')]
for line in lines:
 ln = line.split(',')
 if ln[2] in dist.keys() or ln[1] in dist.keys():
  pass
 else:
  print(ln)
