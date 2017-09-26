import os
import cPickle
import gzip

#database_path = '/mnt/ramdisk/tmp/descriptors/'
database_path = 'tmp/descriptors/'

def clean_path_name(path_name):
  path_name = path_name.replace("/","_")
  path_name = path_name.replace("\\","_")
  return path_name

def clean_thumb(thumbnail_size):
  thumbnail_size = str(thumbnail_size)
  thumbnail_size = thumbnail_size.replace("(","")
  thumbnail_size = thumbnail_size.replace(")","")
  thumbnail_size = thumbnail_size.replace(",","_")
  thumbnail_size = thumbnail_size.replace(" ","")
  return thumbnail_size

def create_filename(image_path, feature_type, descriptor_type, thumbnail_size):
  thumbnail_size = clean_thumb(thumbnail_size)
  path_name = clean_path_name(image_path)
  file_path = ""
  file_path += str(feature_type).lower() + "_"
  file_path += str(descriptor_type).lower() + "_"
  file_path += thumbnail_size + "_"
  file_path += path_name
  file_path += ".gz"
  return file_path

def load_image(image_path, feature_type, descriptor_type, thumbnail_size):
  file_path = create_filename(image_path, feature_type, descriptor_type, thumbnail_size)
  file_path = database_path + file_path
  if not os.path.isfile(file_path):
    return None
  file_p = gzip.GzipFile(file_path, 'rb')
  buffer = ""
  while 1:
    data = file_p.read()
    if data == "":
      break
    buffer += data
  descriptors = cPickle.loads(buffer)
  file_p.close()
  return descriptors

def save_image(image_path, feature_type, descriptor_type, thumbnail_size, descriptors):
  file_path = create_filename(image_path, feature_type, descriptor_type, thumbnail_size)
  file_path = database_path + file_path
  if not os.path.exists(database_path):
    os.makedirs(database_path)
  file_pi = gzip.GzipFile(file_path, 'wb')
  file_pi.write(cPickle.dumps(descriptors, 2))
  file_pi.close()
  print "File saved: " + str(file_path)

def load_image_old(image_path, feature_type, descriptor_type, thumbnail_size):
  file_path = create_filename(image_path, feature_type, descriptor_type, thumbnail_size)
  file_path = database_path + file_path
  if not os.path.isfile(file_path):
    return None
  file_p = open(file_path, 'r') 
  descriptors = cPickle.load(file_p)
  return descriptors

def save_image_old(image_path, feature_type, descriptor_type, thumbnail_size, descriptors):
  file_path = create_filename(image_path, feature_type, descriptor_type, thumbnail_size)
  file_path = database_path + file_path
  if not os.path.exists(database_path):
    os.makedirs(database_path)
  file_pi = open(file_path, 'w') 
  cPickle.dump(descriptors, file_pi)
