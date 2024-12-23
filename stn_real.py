import pandas as pd
import requests
from bs4 import BeautifulSoup
url = "https://www.ogimet.com/cgi-bin/gsynres?lang=en&ind=42182&ndays=30&ano=2024&mes=07&day=01&hora=03&ord=DIR&Send=Send"
dfs = pd.read_html(url)
names = dfs[2].columns
date = dfs[2][('Date', 'Date')]
rain = dfs[2][('Prec. (mm)', 'Prec. (mm)')]
# concatenating the DataFrames 
df = pd.concat([date, rain], join = 'outer', axis = 1)
print(df)
'''

print(dfs[2]['Max'])
print(dfs[2]['Rain'+'\n'+'(mm)'])


df = pd.DataFrame(columns=['Date', 'Rain(mm)'])
df_

#url = "https://www.ogimet.com/cgi-bin/gsynres?lang=en&ind=42182&ndays=30&ano=2024&mes=07&day=01&hora=03&ord=DIR&Send=Send"
html = requests.get(url).text
soup = BeautifulSoup(html,"lxml")
table = soup.find('table')
rows = table.findAll('tr')
for tr in rows:
 cols = tr.findAll('td')
 lst = []
 for td in cols:
  lst.append(td.get_text().encode("utf-8"))
 if len(lst) == 22:
  lst = [x.decode() for x in lst]
 print(lst)
yr = list(range(2024,2025))
frames = []
for y in yr:
 y1 = str(y)
 url = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/"+y1+"/42809099999.csv"
 print('dowloaded.....',url)
 df = pd.read_csv(url)
 df1 = pd.DataFrame.from_dict({'name':df['NAME'],'lat':df['LATITUDE'],'lon':df['LONGITUDE'],'date':df['DATE'].tolist(),'prcp':df['PRCP'].tolist(),'slp':df['SLP'].tolist(),'temp':df['TEMP'].tolist(),'dewp':df['DEWP'].tolist()})
 frames.append(df1)
result = pd.concat(frames)
result.to_csv('synop_42809.csv')
print('done...........')
'''