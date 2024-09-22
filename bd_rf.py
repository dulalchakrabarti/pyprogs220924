import requests
from bs4 import BeautifulSoup
fl = open("bmd.csv","w+")
fl.write("name"+","+"id"+","+"weekday"+","+"dd-mmm"+","+"yyyy_hh_mm_ss"+","+"rate"+","+"acc"+","+"rf"+"\n")
#name	id	weekday	dd-mmm	yyyy_hh_mm_ss	rate	acc	tot
stn_name =['Rajshahi','Dhaka','Dhaka','Barishal','Chattogram','Coxbazar','Dinajpur','Gopalganj','Khulna','Netrokona','Nikili','Nilphamari','Patuakhali','Rangamati','Sreemangal','Sylhet']
stn_list =['18080938','18152957','18153927','18241981','18153762','18150267','18154280','18153231','18153424','18239193','18152105','125416292','18240160','18236381','18236299','18240808']
for num in range(len(stn_list)):
 url = "https://bmdrainfallnetwork.com/realtimedatas/location/"+stn_list[num]
 html = requests.get(url).text
 #print(html)
 soup = BeautifulSoup(html,"lxml")
 table = soup.find('table')
 rows = table.findAll('tr')
 for tr in rows:
  cols = tr.findAll('td')
  lst = []
  for td in cols:
   lst.append(td.get_text().encode("utf-8"))
  lst1 = [x.decode("utf-8") for x in lst]
  if len(lst1) >0:
   #print(stn_name[num],len(lst1))
   fl.write(stn_name[num]+','+lst1[0]+','+str(lst1[1])+' '+str(lst1[2])+','+str(lst1[3])+','+str(lst1[4])+','+str(lst1[5])+'\n')
print("done.......")
'''
  if (len(lst1)) > 0:
  lat = stn[lst1[2]][0]
  lon = stn[lst1[2]][1]
  print(lst1[2],lat,lon,lst1[3],lst1[4],lst1[5])
  fl.write(lst1[2]+','+str(lat)+','+str(lon)+','+str(lst1[3])+' '+str(lst1[4])+','+str(lst1[5])+','+str(lst1[6])+','+str(lst1[9])+','+str(lst1[11])+','+str(lst1[12])+','+str(lst1[13])\
  +','+str(lst1[15])+'\n')
Rajshahi			152			https://bmdrainfallnetwork.com/realtimedatas/location/18080938
Dhaka			53			https://bmdrainfallnetwork.com/realtimedatas/location/18152957
Bagura			44			https://bmdrainfallnetwork.com/realtimedatas/location/18153927
Barishal			220			https://bmdrainfallnetwork.com/realtimedatas/location/18241981
Chattogram						https://bmdrainfallnetwork.com/realtimedatas/location/18153762
Coxbazar			730			https://bmdrainfallnetwork.com/realtimedatas/location/18150267
Dinajpur			937			https://bmdrainfallnetwork.com/realtimedatas/location/18154280
Gopalganj			139			https://bmdrainfallnetwork.com/realtimedatas/location/18153231
Khulna			124			https://bmdrainfallnetwork.com/realtimedatas/location/18153424
Netrokona			108			https://bmdrainfallnetwork.com/realtimedatas/location/18239193
Nikili			232			https://bmdrainfallnetwork.com/realtimedatas/location/18152105
Nilphamari			280			https://bmdrainfallnetwork.com/realtimedatas/location/125416292
Patuakhali			271			https://bmdrainfallnetwork.com/realtimedatas/location/18240160
Rangamati			170			https://bmdrainfallnetwork.com/realtimedatas/location/18236381
Sreemangal			87			https://bmdrainfallnetwork.com/realtimedatas/location/18236299
Sylhet			21			https://bmdrainfallnetwork.com/realtimedatas/location/18240808

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
'''

