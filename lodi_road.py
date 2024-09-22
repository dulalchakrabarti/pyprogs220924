import requests
from bs4 import BeautifulSoup
import tabula
stn ={'LODI ROAD':['28.59','77.2221']}
lat = ''
lon = ''
#read lat long file & store in a dictionary
fl = open("lodi_road.csv","w+")
fl.write("stn"+","+"lat"+","+"lon"+","+"date"+","+"rf"+","+"tt"+","+"rh"+","+"wd"+","+"ws"+","+"gust"+","+"mslp"+"\n")
#url = "http://aws.imd.gov.in:8091/AWS/dataview.php?a=AWS&b=DELHI&c=NEW_DELHI&d=LODI ROAD&e=2019-01-01&f=2019-01-31&g=03&h=00"
url = "http://aws.imd.gov.in:8091/AWS/dataview.php?a=AWS&b=DELHI&c=NEW_DELHI&d=LODI%20ROAD&e=2022-06-01&f=2024-8-15&g=ALL_HOUR&h=ALL_MINUTE"
html = requests.get(url).text
soup = BeautifulSoup(html,"lxml")
table = soup.find('table')
rows = table.findAll('tr')
for tr in rows:
 cols = tr.findAll('td')
 lst = []
 for td in cols:
  lst.append(td.get_text().encode("utf-8"))
 lst1 = [x.decode("utf-8") for x in lst]
 if (len(lst1)) > 0:
  lat = stn[lst1[2]][0]
  lon = stn[lst1[2]][1]
  print(lst1[2],lat,lon,lst1[3],lst1[4],lst1[5])
  fl.write(lst1[2]+','+str(lat)+','+str(lon)+','+str(lst1[3])+' '+str(lst1[4])+','+str(lst1[5])+','+str(lst1[6])+','+str(lst1[9])+','+str(lst1[11])+','+str(lst1[12])+','+str(lst1[13])\
  +','+str(lst1[15])+'\n')
''' 
url = "http://aws.imd.gov.in:8091/AWS/dataview.php?a=AWS&b=UTTAR_PRADESH&c=GAUTAM_BUDDHA_NAGAR&d=ALL_STATION&e=2019-01-01&f=2019-04-01&g=ALL_HOUR&h=ALL_MINUTE"
html = requests.get(url).text
soup = BeautifulSoup(html,"lxml")
table = soup.find('table')
rows = table.findAll('tr')
for tr in rows:
 cols = tr.findAll('td')
 lst = []
 for td in cols:
  lst.append(td.get_text().encode("utf-8"))
 lst1 = [x.decode("utf-8") for x in lst]
 if (len(lst1)) > 0:
  lat = stn[lst1[2]][0]
  lon = stn[lst1[2]][1]
  print(lst1[2],lat,lon,lst1[3],lst1[4],lst1[5]) 
'''