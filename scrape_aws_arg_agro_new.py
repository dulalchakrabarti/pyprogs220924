import requests
from bs4 import BeautifulSoup
import tabula
#read lat long file & store in adictionary
stn = {}
gl = open('aws.csv','w')
lines = [line.rstrip('\n') for line in open('awslatlong.csv')]
for inp in lines:
 lst = inp.split(',')
 if len(lst) > 3:
  stn[lst[3]] = [lst[4],lst[5]]
date = input('Input date(2018-03-29)?')
time = input('Input time in UTC(1)?')
st_lst = ['ANDAMAN_AND_NICOBAR','ANDHRA_PRADESH','ARUNACHAL_PRADESH','ASSAM','BANGLADESH','BHUTAN','BIHAR','CHANDIGARH','CHHATISGARH','DAMAN_AND_DIU','DELHI',
'GOA','GUJARAT','HARYANA','HIMACHAL_PRADESH','JAMMU_AND_KASHMIR','JHARKHAND','KARNATAKA','KERALA','LAKSHADWEEP','MADHYA_PRADESH',
'MAHARASHTRA','MANIPUR','MEGHALAYA','MIZORAM','NAGALAND','NEPAL','ORISSA','PUDUCHERRY','PUNJAB','RAJASTHAN','SIKKIM',
'TAMIL_NADU','TELANGANA','TRIPURA','UTTARAKHAND','UTTAR_PRADESH','WEST_BENGAL']
gl.write('stn'+','+'lat'+','+'lon'+','+'rain'+'\n')
for item in st_lst:
 url = "http://aws.imd.gov.in:8091/AWS/dataview.php?a=aws&b="+item+"&c=ALL_DISTRICT&d=ALL_STATION&e="+date+"&f="+date+"&g="+time+"&h=ALL_MINUTE"
 #print(url)
 html = requests.get(url).text
 soup = BeautifulSoup(html,"lxml")
 table = soup.find('table')
 rows = table.findAll('tr')
 for tr in rows:
  cols = tr.findAll('td')
  lst = []
  for td in cols:
   lst.append(td.get_text().encode("utf-8"))
  if len(lst)>0:
   lst = [x.decode() for x in lst]
   name = lst[2]
   print(lst[:5])
   stn[name].append(lst[5])
for key in stn.keys():
 if len(stn[key]) > 2:
  rain = max(stn[key][2:])
  txt = ','.join(stn[key][:2])
  gl.write(key+','+txt +','+ rain + '\n')
for item in st_lst:
 url = "http://aws.imd.gov.in:8091/AWS/dataview.php?a=arg&b="+item+"&c=ALL_DISTRICT&d=ALL_STATION&e="+date+"&f="+date+"&g="+time+"&h=ALL_MINUTE"
 html = requests.get(url).text
 soup = BeautifulSoup(html,"lxml")
 table = soup.find('table')
 rows = table.findAll('tr')
 for tr in rows:
  cols = tr.findAll('td')
  lst = []
  for td in cols:
   lst.append(td.get_text().encode("utf-8"))
  if len(lst)>0:
   lst = [x.decode() for x in lst]
   name = lst[2]
   print(lst[:5])
   stn[name].append(lst[5])
for key in stn.keys():
 if len(stn[key]) > 2:
  rain = max(stn[key][2:])
  txt = ','.join(stn[key][:2])
  gl.write(key+','+txt +','+ rain + '\n')
for item in st_lst:
 url = "http://aws.imd.gov.in:8091/AWS/dataview.php?a=agro&b="+item+"&c=ALL_DISTRICT&d=ALL_STATION&e="+date+"&f="+date+"&g="+time+"&h=ALL_MINUTE"
 html = requests.get(url).text
 soup = BeautifulSoup(html,"lxml")
 table = soup.find('table')
 rows = table.findAll('tr')
 for tr in rows:
  cols = tr.findAll('td')
  lst = []
  for td in cols:
   lst.append(td.get_text().encode("utf-8"))
  if len(lst)>0:
   lst = [x.decode() for x in lst]
   name = lst[2]
   print(lst[:5])
   stn[name].append(lst[5])
for key in stn.keys():
 if len(stn[key]) > 2:
  rain = max(stn[key][2:])
  txt = ','.join(stn[key][:2])
  gl.write(key+','+txt +','+ rain + '\n')
for item in st_lst:
 url = "http://aws.imd.gov.in:8091/AWS/dataview.php?a=AWSAGRO&b="+item+"&c=ALL_DISTRICT&d=ALL_STATION&e="+date+"&f="+date+"&g="+time+"&h=ALL_MINUTE"
 html = requests.get(url).text
 soup = BeautifulSoup(html,"lxml")
 table = soup.find('table')
 rows = table.findAll('tr')
 for tr in rows:
  cols = tr.findAll('td')
  lst = []
  for td in cols:
   lst.append(td.get_text().encode("utf-8"))
  if len(lst)>0:
   lst = [x.decode() for x in lst]
   name = lst[2]
   print(lst[:5])
   stn[name].append(lst[5])
for key in stn.keys():
 if len(stn[key]) > 2:
  rain = max(stn[key][2:])
  txt = ','.join(stn[key][:2])
  gl.write(key+','+txt +','+ rain + '\n')
tabula.convert_into("rf.pdf", "rf.csv", output_format="csv",pages='all')
lines = [line.rstrip('\n') for line in open('awslatlong_dis.csv')]
for inp in lines:
 lst = inp.split(',')
 #print(lst)
 stn[lst[0]] = [lst[1],lst[2]]
hl = open('rf.csv')
line = hl.readline()
while line:
 line = line.split(',')
 if line[0].isdigit():
  line = [x for x in line if x != '']
  #print(line)
  gl.write(line[1]+','+stn[line[1]][0] +','+stn[line[1]][1]+','+line[2] + '\n')
  print(line[1],stn[line[1]][0],stn[line[1]][1],line[2])
 line = hl.readline()
 line = line.strip('\n')
hl.close()



