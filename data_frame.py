import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
lines = [line.rstrip() for line in open('fi.csv')]
fit = []
fip = []
for line in lines:
 ln = line.split(',')
 fit.append(ln[0])
 fip.append(ln[1])
df = pd.DataFrame({'time':fit[1:],'share':fip[1:]})
tm = df['time'].values.tolist()
tmi = [int(x) for x in tm]
fp = df['share'].values.tolist()
fp1 = [float(x) for x in fp]
lines = [line.rstrip() for line in open('fo.csv')]
fo = {}
count = 0
for line in lines:
 ln = line.split(',')
 fo[count] = ln[1:]
 count+=1
tmo = [ int(x)+144 for x in fo[0]]
fpo = [float(x) for x in fo[1]]
plt.figure(figsize=(14, 7))
plt.plot(tmi,fp1, 'o-b')
plt.plot(tmo,fpo,'o-r')
plt.xlabel("Date")
plt.ylabel("share")
plt.title("Actual")
plt.legend()
plt.show()
print(tmi)
print(tmo)
print(fp1)
print(fpo)
