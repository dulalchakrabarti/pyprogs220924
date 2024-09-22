import pandas as pd
lines = [line.rstrip() for line in open('ncr24.csv')]
r24 = {}
lst = []
for line in lines[1:]:
 line = line.split(',')
 ln = line[0].split('-')
 print(ln)
  