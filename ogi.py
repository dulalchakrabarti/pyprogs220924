import urllib.request
import time
rep_count = 0
fl = open('rain_42724.csv','a')
yrs = list(range(2016,2025))
#yrs = [2024]
for y in yrs:
 y1 = str(y)
 print(y1)
 url = "http://www.ogimet.com/cgi-bin/getsynop?block=42724&begin="+y1+"01010000&end="+y1+"12312300"
 f2 = urllib.request.urlopen(url)
 txt1 = f2.read()
 txt1 = txt1.decode('utf-8')
 txt2 = txt1.split('\n')
 for item in txt2:
  rep_count+=1
  item = item.split(',')
  buf = item[-1].split(' ')
  print(buf)
  if '555' in buf:
   ptr = buf.index('555')
   if buf[ptr+1][0] == '0':
    txt3=item[3]+'-'+item[2]+'-'+item[1]+' '+item[4]+'00'+','+buf[ptr+1][1:]
    fl.write(txt3+'\n')
  else:
    if len(buf) > 5:
     rain = '0000'
     txt3=item[3]+'-'+item[2]+'-'+item[1]+' '+item[4]+'00'+','+rain
     fl.write(txt3+'\n')
     print(txt3)
 print('dowloaded.....',url,rep_count,'.....reports')
 time.sleep(250)
print('done...........')
