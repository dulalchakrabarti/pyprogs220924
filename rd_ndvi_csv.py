import pandas as pd
df = pd.read_csv('ndvi.csv',header = 0)
r,c = df.shape
df = {}
for k,row in df.iterrows():
 lst = row.tolist()