import pandas as pd
yr = list(range(1994,2025))
frames = []
for y in yr:
 y1 = str(y)
 url = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/"+y1+"/42809099999.csv"
 print('dowloaded.....',url)
 df = pd.read_csv(url)
 df1 = pd.DataFrame.from_dict({'date':df['DATE'].tolist(),'prcp':df['PRCP'].tolist(),'slp':df['SLP'].tolist(),'temp':df['TEMP'].tolist(),'dewp':df['DEWP'].tolist()})
 frames.append(df1)
result = pd.concat(frames)
result.to_csv('synop_42809.csv')
print('done...........')
