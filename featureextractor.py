import cv2
from PIL import Image
import numpy
import skimage
from skimage.feature import daisy, hog
import subprocess
import overfeat
import numpy
from scipy.ndimage import imread
from scipy.misc import imresize
import database_file
import time
from find_obj import init_feature
from asift import affine_detect

print "Initializing overfeat"
overfeat.init('overfeat/overfeat/data/default/net_weight_0', 0)


#overfeat_initialized = False
def extract_overfeat(image_path):
  # if overfeat_initialized == None or not overfeat_initialized:
    
  #   overfeat_initialized = True
  print "Overfeat: ", image_path
  image = imread(image_path)
  print "Image shape: ", image.shape
  if len(image.shape) == 2 or image.shape[2] == 2:
    image = skimage.color.gray2rgb(image)
  elif image.shape[2] == 4:
    image_rgb = numpy.zeros((image.shape[0],image.shape[1], 3), numpy.uint8)
    image_rgb[:,:,0] = image[:,:,0]
    image_rgb[:,:,1] = image[:,:,1]
    image_rgb[:,:,2] = image[:,:,2]
    image = image_rgb
  h0 = image.shape[0]
  w0 = image.shape[1]
  d0 = float(min(h0, w0))
  image = image[int(round((h0-d0)/2.)):int(round((h0-d0)/2.)+d0),
              int(round((w0-d0)/2.)):int(round((w0-d0)/2.)+d0), :]
  image = imresize(image, (231, 231)).astype(numpy.float32)
  #image = cv2.resize(image, (231, 231)).astype(numpy.float32)
  # numpy loads image with colors as last dimension, transpose tensor
  h = image.shape[0]
  w = image.shape[1]
  c = image.shape[2]
  image = image.reshape(w*h, c)
  image = image.transpose()
  image = image.reshape(c, h, w)
  print "Image size :", image.shape
  out_categories = overfeat.fprop(image)
  #layer 21,22,23
  layer_output = overfeat.get_output(20)
  print "Layer size: ", layer_output.shape
  layer_output = layer_output.flatten()
  descriptors = []
  descriptors.append(layer_output)
  out_categories = out_categories.flatten()
  top = [(out_categories[i], i) for i in xrange(len(out_categories))]
  top.sort()
  print "\nTop classes :"
  for i in xrange(5):
      print(overfeat.get_class_name(top[-(i+1)][1]))
  
  return descriptors

def extract_descriptor(image_path, thumbnail_size, feature_type="SIFT", descriptor_type="SIFT"):
  t_start_cached = time.time()
  descriptors_db = database_file.load_image(image_path, feature_type, descriptor_type, thumbnail_size)
  t_end_cached = time.time() - t_start_cached
  if descriptors_db == None:
    print "Precomputing: " + str(image_path)
    t_start = time.time()
    features, descriptors = extract_descriptor_compute(image_path, thumbnail_size, feature_type, descriptor_type)
    database_file.save_image(image_path, feature_type, descriptor_type, thumbnail_size, descriptors)
    t_end = time.time() - t_start
    print "Desc time: " + str(t_end)
    return None, descriptors
  else:
    print "Cached: " + str(image_path)
    print "Desc cached: " + str(t_end_cached)
    return None, descriptors_db

def extract_descriptor_compute(image_path, thumbnail_size, feature_type="SIFT", descriptor_type="SIFT"):

  if feature_type == "OVERFEAT" or descriptor_type == "OVERFEAT":
    descriptor = extract_overfeat(image_path)
    return None, descriptor

  img = Image.open(image_path)
  img.load()

  if img == None:
    raise Exception("Error: Cannot read " + image_path) 
  if img.size > thumbnail_size:
    img.thumbnail(thumbnail_size)
  img_array = numpy.array(img)
  return extract_descriptor_process(img_array, thumbnail_size, feature_type, descriptor_type)


def extract_descriptor_process(image_array, thumbnail_size, feature_type="SIFT", descriptor_type="SIFT"):
  

  #Converte para cinza se tiver mais que 3 canais (RGB)
  img_gray = image_array
  if len(image_array.shape) >= 3:
    img_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
  img_gray = numpy.array(img_gray, numpy.uint8)

  if descriptor_type == "DAISY":
    try:
      daisy_descs = []
      descs = daisy(img_gray, step=15, radius=15, rings=14, histograms=6, orientations=8)
      for row in range(descs.shape[0]):
        for col in range(descs.shape[1]):
          daisy_descs.append(descs[row][col])

      #print len(daisy_descs)
      return None, daisy_descs
    except:
      print "Error while extracting Daisy descriptor"
      return None, None

  if descriptor_type == "HOG":
    descs = hog(img_gray, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
    if len(descs) < 512:
      return None, None

    hog_descs = []
    hog_descs.append(descs[0:648])
    return None, hog_descs     

  if feature_type == "ASIFT" or descriptor_type == "ASIFT":
    detector, matcher = init_feature('sift-flann')
    kp, descs = affine_detect(detector, img_gray)
    return kp, descs

  #Normalize
  #img_gray = img_gray.astype(numpy.float32)
  #img_gray = (img_gray - img_gray.mean())/img_gray.std()
  #img_gray = cv2.normalize(img_gray, img_gray,0, 255, cv2.NORM_MINMAX)
  #img_gray = img_gray.astype(numpy.uint8)
  
  featureDetector = cv2.FeatureDetector_create(feature_type)
  descriptorDetector = cv2.DescriptorExtractor_create(descriptor_type)
  
  keypoints = featureDetector.detect(img_gray)
  keypoints, descriptors = descriptorDetector.compute(img_gray, keypoints)
  
  return keypoints, descriptors

def extract_features_spatial(image_path, thumbnail_size, feature_type="SURF", descriptor_type="SURF"):
  img = Image.open(image_path)
  img.load()
  
  if img == None:
    raise Exception("Error: Cannot read " + image_path) 
  if img.size > thumbnail_size:
    img.thumbnail(thumbnail_size)
  img_array = numpy.array(img)
  #Converte para cinza se tiver mais que 3 canais (RGB)
  img_gray = img_array
  if len(img_array.shape) >= 3:
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
  img_gray = numpy.array(img_gray, numpy.uint8)

  p1 = img_gray[0:img_gray.shape[0]/2, 0:img_gray.shape[1]/2]
  p2 = img_gray[0:img_gray.shape[0]/2, img_gray.shape[1]/2:]
  p3 = img_gray[img_gray.shape[0]/2:0, 0:img_gray.shape[1]/2]
  p4 = img_gray[img_gray.shape[0]/2:0, img_gray.shape[1]/2:0]
  patches = [p1, p2, p3, p4]
  featureDetector = cv2.FeatureDetector_create(feature_type)
  descriptorDetector = cv2.DescriptorExtractor_create(descriptor_type)
  patches_keypoints = []
  patches_descriptors = []
  for patch in patches:
    keypoints = featureDetector.detect(patch)
    keypoints, descriptors = descriptorDetector.compute(patch, keypoints)
    patches_keypoints.append(keypoints)
    patches_descriptors.append(descriptors)
      
  
  return patches_keypoints, patches_descriptors

def extract_ref2(image_path, thumbnail_size, num_subimages):
  img = Image.open(image_path)
  img.load()
  if img == None:
    raise Exception("Error: Cannot read " + image_path) 
  if img.size > thumbnail_size:
    img.thumbnail(thumbnail_size)
  img_array = numpy.array(img)
  img_gray = img_array
  if len(img_array.shape) >= 3:
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
  img_gray = numpy.array(img_gray, numpy.uint8)
  
  p1 = img_gray[0:img_gray.shape[0]/2, 0:img_gray.shape[1]/2]
  p2 = img_gray[0:img_gray.shape[0]/2, img_gray.shape[1]/2:]
  p3 = img_gray[img_gray.shape[0]/2:0, 0:img_gray.shape[1]/2]
  p4 = img_gray[img_gray.shape[0]/2:0, img_gray.shape[1]/2:0]
  patches = [p1, p2, p3, p4]
  patch_total_hist = []
  for patch in patches:
    featureDetector = cv2.FeatureDetector_create("SURF")
    keypoints = featureDetector.detect(img_gray)
    kp_angles = []
    if keypoints == None:
     patch_hist = numpy.zeros((35),numpy.float32)
     print "Patch hist: ", len(patch_hist)
     for hist_item in patch_hist:
      patch_total_hist.append(hist_item)
     continue

    for keypoint in keypoints:
      kp_angles.append(keypoint.angle)
    bins = range(0, 360, 10)
    patch_hist = numpy.histogram(kp_angles, bins=bins, density=True)[0]
    
    for hist_item in patch_hist:
      patch_total_hist.append(hist_item)

  patch_total_hist = numpy.asarray(patch_total_hist, numpy.float32)
  return patch_total_hist

def extract_ref(image_path, thumbnail_size):
  img = Image.open(image_path)
  img.load()
  if img == None:
    raise Exception("Error: Cannot read " + image_path) 
  if img.size > thumbnail_size:
    img.thumbnail(thumbnail_size)
  img_array = numpy.array(img)
  img_gray = img_array
  if len(img_array.shape) >= 3:
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
  img_gray = numpy.array(img_gray, numpy.uint8)
  featureDetector = cv2.FeatureDetector_create("SURF")
  keypoints = featureDetector.detect(img_gray)
  kp_angles = []
  if keypoints == None:
    return []

  for keypoint in keypoints:
    kp_angles.append(keypoint.angle)
  bins = range(0, 360, 10)
  histogram = numpy.histogram(kp_angles, bins=bins, density=True)[0]
  return histogram




