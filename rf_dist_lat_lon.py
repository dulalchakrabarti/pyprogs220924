fl = open('dist_name_lat_lon.csv','w+')
lins = [lns.rstrip('\n') for lns in open('dist.csv')]
for ln in lins:
 ln = ln.split(',')
 print(ln[3].lower(),ln[4],ln[5])
 fl.write(ln[3].lower()+','+ln[4]+','+ln[5]+'\n')