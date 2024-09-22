import pandas as pd
fl = open('ser.csv','w+')
lines = [line.rstrip() for line in open('delhinoida.csv')]
ncr = {}
for line in lines:
 line = line.split(',')
 print(line)
 if len(line) > 7:
  #print(line[4],line[0],line[1],line[5].split('-'),line[6].split(':'),line[7])
  #print(line[4],line[0],line[1],line[5],line[6],line[7])
  fl.write(line[4]+','+line[0]+','+line[1]+','+line[5]+','+line[6]+','+line[7]+'\n')
  ncr[line[4]+'-'+line[0]+'-'+line[1]+'-'+line[5]+'-'+line[6]] = [line[5],line[6],line[7]]
  #print(line[4]+'_'+line[0]+'_'+line[1]+'_'+line[5]+'_'+line[6],line[5],line[6],line[7])
print(ncr.keys())

df = pd.DataFrame.from_dict(ncr)
df1 = df.T
df1.to_csv('ncr24.csv')
print(df1.head)

