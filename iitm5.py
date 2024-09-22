import pandas as pd
import matplotlib.pyplot as plt
lines = [line.rstrip() for line in open('iitm_rf.txt')]
ser = {}
for line in lines[1:]:
 line = line.split()
 ser[line[0]] = line[1:13]
keylist=ser.keys()
sorted(keylist)
sr = []
for key in keylist:
 sr.extend(ser[key])
x = [0,1,2,3,4,5,6,7,8,9]
y = sr[:10]
print(x)
print(y)
plt.plot(x,y)
plt.show()