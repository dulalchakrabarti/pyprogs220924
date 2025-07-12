import os
import glob
import pandas as pd
from PIL import Image
frmr = {}
count =  0
base = '/media/dc/DC-28122024/wheatcam1/'
base_ = '/media/dc/DC-28122024/wheatcam7/'
files = glob.glob(base+'/*.JPG')
for file in files:
 fl = file.split('/')
 img = fl[-1].split('_')
 fmrid = img[0]
 fldid = img[1].split('.')
 #key = fmrid+'_'+fldid[0]
 key =fmrid
 if key in frmr.keys():
  frmr[key].append([file])
 else:
  frmr[key] = [file]
df = pd.read_csv('Actual_Yields_CCE_2018.csv')
lst1 = df['farmer_id'].tolist()
lst2 = df['site_id'].tolist()
lst3 = df['yield_avg'].tolist()
l = [[i,j,k] for i,j,k in zip(lst1,lst2,lst3)]
keylist = frmr.keys()
sorted(keylist)
for key in keylist:
 for item in l:
  if int(item[0]) == int(key):
   subfolder_name = base_+str(item[0])+'_'+str(item[1])
   yld = round(item[2])
   # Create the subfolder
   # exist_ok=True prevents an error if the directory already exists
   os.makedirs(subfolder_name, exist_ok=True) 
   nested_list = frmr[key]
   for item in nested_list:
    if isinstance(item,list):
     img = Image.open(item[0])
     # Define the new dimensions (width, height)
     new_size = (224, 224)
     # Resize the image
     resized_img = img.resize(new_size)
     dest_path = os.path.join(subfolder_name+"/", str(yld)+".jpg")
     resized_img.save(dest_path)
    else:
     img = Image.open(item)
     # Define the new dimensions (width, height)
     new_size = (224, 224)
     # Resize the image
     resized_img = img.resize(new_size)
     dest_path = os.path.join(subfolder_name+"/", str(yld)+".jpg")
     resized_img.save(dest_path)
   count+=1
print(count)


