import datetime
import pandas as pd
import calendar
import random
fl = open('hist_month.csv','w+')
fl.write('unique_id'+','+'ds'+','+'y'+'\n')

months = []
for y in range(1975, 1987):
    for m in range(1, 13):
        if (y == 2010) and m < 8:
            continue
        if (y == 2016) and m > 2:
            continue
        month = '%s-%s-01' % (y, ('0%s' % (m)) if m < 10 else m)
        months.append(month)
dates_ = [date.split('-') for date in months]
dates = [datetime.datetime(int(date[0]),int(date[1]),int(date[2])) for date in dates_]
month_last_dates = [datetime.datetime(date.year, date.month,
      calendar.monthrange(date.year, date.month)[1]) for date in dates]
mon = []
# months
for date in month_last_dates:
  mon.append(str(date.year)+'-'+str( date.month)+'-'+str(date.day))
m = len(mon)
lst=[]
for n in range(m):
 num = random.uniform(682.0,1187650.0)
 num = round(num,1)
 lst.append(num)
uid = []
for y_ in range(12):
 for k in range(12):
  uid_ = 'T'+str(int(y_))
  uid.append(uid_)
for id,ds,y in zip(uid,mon,lst):
 unique_id = id
 fl.write(unique_id+','+ds+','+str(y)+'\n')
