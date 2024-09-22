# Use case for downloading wmo station indexes from ogimet
# Author: Dulal Chakrabarti
# Date:27.06.2022
#dc@dc-Len:~/codes$ python3 ogi_one.py  - command to run ogi_one
#Enter start datetime(201807090300):202206270300
#Enter end datetime(201807090359):202206270359
#42473,2022,06,27,03,00,AAXX 27034 42473 32996 00000 10336 20279 40006 333 10404 20/// 58022 555 10016= - data for Banda

#42273,2022,06,27,03,00,AAXX 27034 42273 31595 61404 10322 20251 40032 70522 84530 333 10374 20/// 58011 84625 86360 555 10059= - Data for Bahraich which is near Gonda
#
from pymetdecoder import synop as s
import requests # python module for downloading data from websites
beg = ''
end = ''
beg = input('Enter start datetime(201807090300):')
end = input('Enter end datetime(201807090359):')
blk = ['42182'] #banda,bahraich near gonda# you can add to this list more wmo station id
for blknum in blk:
 txt = "http://www.ogimet.com/cgi-bin/getsynop?block=blknum&begin=200912010000&end=200912040000
 txt = "https://www.ogimet.com/cgi-bin/getsynop?begin="+beg+"&end="+end+"&block="+blknum
 r1 = requests.get(txt)
 resp = r1.text
 resp = resp.split(',')
 resp = resp[-1].split('=')
 synop = resp[0]
 output = s.SYNOP().decode(synop)
 print(output)
