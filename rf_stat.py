import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
fl = open('dist_lat_lon.csv')
lines = [line.rstrip('\n') for line in fl]
countd = 0
countn = 0
counte = 0
dist = {}
lins = [lns.rstrip('\n') for lns in open('dist.csv')]
for lin in lins:
 txt = lin.split(',')
 dist[txt[3]] = [txt[4],txt[5]]
for line in lines:
 line = line.split(',')
 print(line)
 #if line[1] in dist.keys():
 if line[2] in dist.keys():
  #print(line)
  #if 'D' in line[-3:] or 'LD' in line[-3:]:
  if 'D' in line[-2:] or 'LD' in line[-2:]:
   countd+=1
   dist[line[2]].append(line[-2:])
  #elif 'N' in line[-3:]:
  elif 'N' in line[-2:]:
  #print(line)
   countn+=1
   dist[line[2]].append(line[-2:])
  #elif 'E' in line[-2:] or 'LE' in line[-2:]:
  elif 'E' in line[-2:] or 'LE' in line[-2:]:
   #print(line)
   counte+=1
   dist[line[2]].append(line[-2:])
  else:
   print(line)
 elif line[1] in dist.keys():
  if 'D' in line[-2:] or 'LD' in line[-2:]:
   countd+=1
   dist[line[1]].append(line[-2:])
  elif 'N' in line[-2:]:
   countn+=1
   dist[line[1]].append(line[-2:])
  elif 'E' in line[-2:] or 'LE' in line[-2:]:
   counte+=1
   dist[line[1]].append(line[-2:])
  else:
   print(line)
keylist = dist.keys()
sorted(keylist)
print(dist)
gl = open('qa.csv','w+')
for key in keylist:
 #print(key,dist[key],len(dist[key]))
 if len(dist[key]) == 3:
  per = dist[key][2][0]
  per = per.split('%')
  per = per[0]
  print(key+','+dist[key][0]+','+dist[key][1]+','+per+','+dist[key][2][1])
  gl.write(key+','+dist[key][0]+','+dist[key][1]+','+per+','+dist[key][2][1]+'\n')
tot = float(countd+countn+counte)
print(countd,countn,counte,tot)
print('-------------------------------------------------------------------------------')
print('deficient','normal','excess','defcount','normalcount','excesscount','totaldist')
print(round((countd*100)/tot),round((countn*100)/tot),round((counte*100)/tot),countd,countn,counte,tot)
