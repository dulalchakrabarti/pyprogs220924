from geopy.geocoders import Nominatim
import tabula
import pandas as pd
url = 'https://cpcb.nic.in//upload/Downloads/AQI_Bulletin_20241102.pdf'
dfs = tabula.io.read_pdf(url, pages='all')
pages = len(dfs) - 1
dfs1 = dfs[:pages]
aq = {}
for item in dfs1:
 city = item['City'].tolist()
 index = item['Index\rValue'].tolist()
 x =zip(city,index)
 for item in tuple(x):
  aq[item[0]] = str(item[1])
keylist = aq.keys()
sorted(keylist)
fl = open('air_quality.csv','w+')
fl.write('city'+','+'lat'+','+'lon'+','+'aq'+'\n')
geolocator = Nominatim(user_agent='dulalchakrabarti@gmail.com')
for key in keylist:
 name = key
 num = aq[key]
 try:
  location = geolocator.geocode(name)
  if location:
   if (location.latitude > 0.0 and location.latitude < 40.0 and location.longitude > 60.0 and location.longitude < 100.0):
    print(name,location.latitude, location.longitude,num)
    fl.write(name+','+str(location.latitude)+','+str(location.longitude)+','+num+'\n' )
 except:
   print('Not found....'+name)

 

