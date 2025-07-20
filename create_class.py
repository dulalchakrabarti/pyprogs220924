import os
from PIL import Image
def list_files_recursively(directory_path):
    """
    Lists the full paths of all files recursively within a given directory.

    Args:
        directory_path (str): The path to the directory to traverse.

    Returns:
        list: A list containing the full paths of all files found.
    """
    file_names = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_names.append(os.path.join(root, file))
    return file_names

# Example usage:
# Replace 'your_directory_path' with the actual path you want to scan
target_directory = '/home/dc/cntk/wheatcam16072025/'
all_files = list_files_recursively(target_directory)

classes = {}
for file_path in all_files:
    yld = file_path.split('_')
    #yld_ = yld.split('.')
    print(yld)
'''
if yld_ in classes.keys():
      classes[yld1_].append(file_path)
     else:
      classes[yld1_] = [file_path]
keylist = classes.keys()
sorted(keylist)
for key in keylist:
 for item in classes[key]:
  item_ = item.split('/')
  yl = item_[-1].split('_')
  if len(yl) == 1:
   txt = yl[0].split('.')
   fld = txt[0]
  else:
   fld = yl[0]
  #print(fld)
  try:
   os.makedirs('/home/dc/cntk/out_img/'+fld, exist_ok=True)
   img = Image.open(item)
   name = item.split('/')
   img.save('/home/dc/cntk/out_img'+fld+'/'+name[-1])
   print(name[-1],'...saved')
  except:
   print('error...subfolder creation')
   pass
'''
