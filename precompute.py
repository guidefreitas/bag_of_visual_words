import featureextractor
import filesprocess
import database
import multiprocessing
import itertools



def precompute(train_path, test_path, feature_type, descriptor_type, thumbnail_size, pool=None):
  print "Precomputing train path: " + str(train_path)
  precompute_compute(train_path, feature_type, descriptor_type, thumbnail_size, pool)
  print "Precomputing test path: " + str(test_path)
  precompute_compute(test_path, feature_type, descriptor_type, thumbnail_size, pool)

def extract(image_info, params):
  image_path = image_info[0]
  feature_type = params[0]
  descriptor_type = params[1]
  thumbnail_size = params[2]
  f,d = featureextractor.extract_descriptor(image_path, thumbnail_size, feature_type, descriptor_type)

def precompute_compute(images_path, feature_type, descriptor_type, thumbnail_size, pool=None):
  images_info = filesprocess.get_files(images_path)
  if pool == None:
    pool = multiprocessing.Pool()
  pool.map(func_star_extract, 
                          itertools.izip(images_info, 
                          itertools.repeat((feature_type, descriptor_type, thumbnail_size))))

  # for image_info in images_info:
  #   image_path = image_info[0]
  #   f,d = featureextractor.extract_descriptor(image_path, thumbnail_size, feature_type, descriptor_type)

def func_star_extract(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return extract(*a_b)