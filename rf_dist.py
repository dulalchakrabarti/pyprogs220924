import tabula
from geopy.geocoders import Nominatim
import time
stn = {}
lines = [line.rstrip() for line in open('dist.csv')]
for line in lines:
 line = line.split(',')
 stn[line[3]] = [line[4],line[5]]
sub={}
lines = [line.rstrip() for line in open('metsub.csv')]
for line in lines:
 line = line.split(',')
 sub[line[2]] = []
print(sub.keys())
fl = open('rf_today.csv','w')
lines = [line.rstrip('\n') for line in open('rf.csv')]
for inp in lines:
 lst = inp.split(',')
 if lst[0].isdigit():
  rf = lst[2]
  if lst[1] != '' and lst[1] == 'LAKSHADWEEP':
   print(lst[1],str(stn[lst[1]][0])+','+str(stn[lst[1]][1]),rf)
   fl.write(str(lst[1])+','+str(stn[lst[1]][0])+','+str(stn[lst[1]][1])+','+str(rf)+'\n')
  else:
   name = lst[1]
   if name not in sub.keys():
    print(name,str(stn[name][0])+','+str(stn[name][1]),rf)
    fl.write(name+','+str(stn[name][0])+','+str(stn[name][1])+','+str(rf)+'\n')
print('done....')
'''
  rf = lst[3]
  if lst[1] != '' and lst[1] == 'LAKSHADWEEP':
   print(lst[1],str(stn[lst[1]][0])+','+str(stn[lst[1]][1]),rf)
   fl.write(str(lst[1])+','+str(stn[lst[1]][0])+','+str(stn[lst[1]][1])+','+str(rf)+'\n')
  else:
   name = lst[1]
   if name not in sub.keys():
    print(name,str(stn[name][0])+','+str(stn[name][1]),rf)
    fl.write(name+','+str(stn[name][0])+','+str(stn[name][1])+','+str(rf)+'\n')
print('done....')
'''
 


