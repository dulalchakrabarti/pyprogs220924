frmr = {}
lines1 = [line for line in open('Actual_Yields_CCE_2018.csv')]
lines2 = [line for line in open('frmr_id_fld.csv')]
for line1 in lines1[1:]:
 ln1 = line1.strip().split(',')
 fid = ln1[0]
 site = ln1[1]
 yld = str(int(float(ln1[2])))
 key = fid+'_'+site
 if key in frmr.keys():
  frmr[key].append(yld)
 else:
  frmr[key] = [yld]
#print(len(frmr))
count = 0
for line2 in lines2:
 ln2 = line2.strip().split(',')
 if 'file' in ln2:
  count+=1
  ln2_ = ln2[1].split('/')
  print(ln2_[-2],ln2_[-1].split('_'))
print(count)
