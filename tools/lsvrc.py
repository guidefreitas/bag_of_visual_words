from PIL import Image
import os
import xml.etree.ElementTree as ET
import filesprocess
import featureextractor
import serializer
import cPickle as pickle
import gzip

def process_directory(dir_to_process, base_cropped_dir):

  files_tuple = []
  for (dirpath, dirnames, filenames) in os.walk(dir_to_process):
    for filename in filenames:
      if filename.endswith('JPEG'):
        path, folder = os.path.split(dirpath)
        files_tuple.append(os.path.join(dirpath, filename))

  for idx, file_info in enumerate(files_tuple):
    print "Processing ", idx+1, " of ", len(files_tuple)
    cropped_info = read_image(file_info)
    save_cropped_images(base_cropped_dir,cropped_info)
  
  return files_tuple

def read_image(path):
  img = Image.open(path)
  path_without_ext, ext = os.path.splitext(path)
  path_parts = os.path.split(path_without_ext)
  path_xml = os.path.join(*path_parts) + '.xml'
  boxes = read_xml(path_xml)

  img_crops = []
  for box in boxes:
    filename = box[0]
    name = box[1]
    xmin = int(box[2])
    ymin = int(box[3])
    xmax = int(box[4])
    ymax = int(box[5])
    cropped = img.crop((xmin,ymin,xmax,ymax))
    img_crops.append((filename, name, cropped))
  return img_crops

def read_xml(path):
  bounding_boxes = []
  tree = ET.parse(path)
  root = tree.getroot()
  filename = root.find('filename').text
  for objectrecog in root.iter('object'):
    name = objectrecog.find('name').text
    bndbox = objectrecog.find('bndbox')
    xmin = bndbox.find('xmin').text
    ymin = bndbox.find('ymin').text
    xmax = bndbox.find('xmax').text
    ymax = bndbox.find('ymax').text
    bounding_boxes.append((filename,name,xmin,ymin,xmax,ymax))
  return bounding_boxes

def save_cropped_images(base_path, crops_info):
  for idx, info in enumerate(crops_info):
    filename = info[0]
    category_name = info[1]
    cropped_image = info[2]
    full_path = base_path + "/" + category_name
    if not os.path.exists(full_path):
      os.makedirs(full_path)
    img_path = full_path + "/" + filename + "_" + str(idx) + ".jpg"
    print "Saving: ", img_path
    cropped_image.save(img_path, "JPEG")

def generate_descriptors_file(path_to_process, path_to_result, batch_size, prefix="train_"):
  print "Generating descriptors file"
  if not os.path.exists(path_to_result):
      os.makedirs(path_to_result)
  images_info = filesprocess.get_files(path_to_process)
  print str(len(images_info)), " files to process."
  count_descriptors = 0
  count_files = 0;
  buffer = []
  for idx, image_info in enumerate(images_info):
    print "Processing ", idx, " of ", len(images_info)
    count_descriptors = count_descriptors + 1
    img_path = image_info[0]
    img_folder = image_info[1]
    f,d = featureextractor.extract_sift(img_path)
    if (d != None):
      p_temp = serializer.pickle_keypoints(f, d)
      buffer.append((img_path, img_folder, p_temp))
    
    if count_descriptors >= batch_size:
      file_name = prefix + str(count_files) + ".des.gz"
      file_path = path_to_result + "/" + file_name
      print "Saving file: ", file_path
      save_desc_file(buffer, file_path)
      count_files += 1
      count_descriptors = 0
      buffer = []

  return images_info

def save_desc_file(data, file_path):
  f = gzip.open(file_path, 'wb')
  data = pickle.dumps(data) #open(file_path, "wb")
  f.write(data)
  f.close()
      
def load_desc_file(file_path):
  f = gzip.open(file_path, 'rb')
  data_descs = pickle.loads(f.read())
  images_data = []
  for data in data_descs:
    img_path = data[0]
    img_folder = data[1]
    data_kp_dsc = data[2]
    kp, dsc = serializer.unpickle_keypoints(data_kp_dsc)
    images_data.append((img_path, img_folder, kp, dsc))
  return images_data

def get_desc_files_list(dir_path):
  files_info = []
  for (dirpath, dirnames, filenames) in os.walk(dir_path):
    for filename in filenames:
      if filename.endswith('.des.gz'):
        files_info.append(dirpath + "/" + filename)
  return files_info
      

