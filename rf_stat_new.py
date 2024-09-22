import pandas as pd
from geopy.geocoders import Nominatim
import time
import tabula
import math
metsub = {}
lins = [lns.rstrip('\n') for lns in open('metsub.csv')]
for lin in lins:
 txt = lin.split(',')
 metsub[txt[2]] = txt[0]
#print(metsub.keys())
lines =  [line.rstrip('\n') for line in open('rf.csv')]
fl = open('dist_lat_lon.csv','w+')
for line in lines:
 ln = line.split(',')
 if ln[0].isdigit():
  if ln[2] not in metsub.keys():
   print(ln)
   fl.write(str(ln[0])+','+str(ln[1])+','+str(ln[2])+','+str(ln[3])+','+str(ln[4])+','+str(ln[5])+','+str(ln[6])+','+str(ln[7])+','+str(ln[8])+','+str(ln[9])+','+str(ln[10])+'\n')
fl.close()