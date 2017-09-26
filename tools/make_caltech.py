import os, sys
import shutil
import numpy

train_path = sys.argv[1]
test_path = sys.argv[2]
percentage = float(sys.argv[3])

print "Train: ", train_path
print "Test: ", test_path


files_dict = dict()
for (dirpath, dirnames, filenames) in os.walk(train_path):
  for filename in filenames:
    if filename.endswith('.jpg'):
      path, folder = os.path.split(dirpath)
      folder = folder.replace('/','')
      if folder in files_dict:
        files_dict[folder].append((os.path.join(dirpath, filename), filename))
      else:
        files_dict[folder] = []
        files_dict[folder].append((os.path.join(dirpath, filename), filename))

for key in files_dict:
  category_data = files_dict[key]
  cat_size = len(category_data)
  reduced_size = int(cat_size * percentage) 
  rand = numpy.random.permutation(cat_size)
  for i in rand[0:reduced_size]:
    img_info = category_data[i]
    img_path = img_info[0]
    img_filename = img_info[1]
    test_category_path = test_path + "/" + key + "/"
    test_img_path = test_category_path + img_filename
    if not os.path.exists(test_category_path):
      os.makedirs(test_category_path)
    print "Move: ", img_path, " to ", test_img_path
    shutil.move(img_path, test_img_path)
    


# for img_data in files_tuple:
#   img_path = img_data[0]
#   folder = img_data[1]
#   filename = img_data[2]
#   test_category_path = test_path + "/" + folder + "/"
#   test_img_path = test_category_path + filename
#   if not os.path.exists(test_category_path):
#     os.makedirs(test_category_path)
#   shutil.move(img_path, test_img_path)
