import cv2
from PIL import Image
import numpy
import libopf_py
import filesprocess
import featureextractor
import argparse
from scipy.spatial import distance
import sys
import hashlib


def distance(desc_a, desc_b):
  desc_a = numpy.asarray(desc_a)
  desc_b = numpy.asarray(desc_b)
  result = numpy.abs(desc_a - desc_b)
  result = numpy.sum(result)
  return result

def desc_hash(desc):
  hash = ""
  for d in desc:
    hash = hash + str("%0.4f" % d )
  dig = hashlib.md5(hash).hexdigest()
  return str(dig)


parser = argparse.ArgumentParser(prog='python demo_opf_unsup.py', usage='%(prog)s [options]',
    description='', 
    epilog="")
parser.add_argument('-dir', '--directory', help='images directory', required=True)

args = parser.parse_args()
print "Processing: " + str(args.directory)
images_info = filesprocess.get_files(args.directory)
print str(len(images_info)) +  " images found."

# all_classes = []
# for idx, image_info in enumerate(images_info):
#   img_folder = image_info[1]
#   all_classes.append(img_folder)

# all_classes = numpy.asarray(all_classes)
# all_classes = numpy.unique(all_classes)
# print str(len(all_classes)) + " classes found."

all_descriptors = []
for idx, image_info in enumerate(images_info):
  img_path = image_info[0]
  img_class = image_info[1]
  features, descriptors = featureextractor.extract_sift(img_path, (256,256),"SIFT","SIFT")
  if descriptors != None:
    for desc in descriptors:
      all_descriptors.append(desc)

#all_descriptors = all_descriptors[0:100]
print "Computing hashes"
hashed_descriptors = {}
for desc in all_descriptors:
  d_hash = desc_hash(desc)
  hashed_descriptors[d_hash] = desc

print "Initializating..."
roots = {}
for hash_idx, key in enumerate(hashed_descriptors):
  print "Roots: " + str(len(roots))
  print "Processing " + str(hash_idx) + " of " + str(len(hashed_descriptors))
  if len(roots) == 0:
    roots[key] = []
    continue

  if len(roots) == 1 and len(roots[roots.keys()[0]]) == 0:
    for key_root in roots:
      desc = hashed_descriptors[key]
      dist = distance(hashed_descriptors[key_root], desc)
      roots[key_root].append({"descriptor": desc, "dist": dist})
  
  else:
    desc = hashed_descriptors[key]
    #get min root
    min_root = ["", sys.float_info.max]
    for key_root in roots:
      desc_root = hashed_descriptors[key_root]
      dist = distance(desc, desc_root)
      #print "Distance: " + str(dist)
      if dist < min_root[1]:
        min_root = [key_root, dist]

    #add descriptor to min root
    data = {"descriptor":desc, "dist":min_root[1]}
    roots[min_root[0]].append(data)

    #find max dist desc
    max_ele = {"descriptor":None, "dist":0.0, "idx":0}
    for idx, ele in enumerate(roots[min_root[0]]):
      ele_hash = desc_hash(ele["descriptor"])
      if ele_hash == key:
        continue
      if ele["dist"] > min_root[1]:
        if ele["dist"] > max_ele["dist"]:
          ele["idx"] = idx
          max_ele = ele

    #print "Max element distance: " + str(max_ele["dist"])
    #print "Max element idx: " + str(max_ele["idx"])
    
    #find min root for the realocated element
    min_root_2 = ["", sys.float_info.max]
    if max_ele["descriptor"] != None:
      #remove max element from old root

      del roots[min_root[0]][max_ele["idx"]]

      for key_root in roots:
        desc_root = hashed_descriptors[key_root]
        dist = distance(desc_root, max_ele["descriptor"])
        if dist < min_root_2[1]:
          min_root_2 = [key_root, dist]

    if min_root_2[1] < max_ele["dist"]:
      max_ele["dist"] = min_root_2[1]
      roots[min_root_2[0]].append(max_ele)
    elif min_root_2[1] != sys.float_info.max:
      new_root_hash = desc_hash(max_ele["descriptor"])
      roots[new_root_hash] = []

print "Roots: " + str(len(roots))
for idx, key in enumerate(roots):
  print str(idx) + " - " + str(len(roots[key]))
#descriptors = descriptors[0:1000]
# dist = distance.squareform(distance.pdist(descriptors, 'sqeuclidean'))
# dist = numpy.ascontiguousarray(dist, numpy.float64)
# print dist.shape
# O = libopf_py.OPF()
# print "Fit OPF"
# O.fit(descriptors)
# pred = O.predict(descriptors)
# print pred[0]
# pred = O.predict(descriptors)
# print pred[0]




# root[[1,2,3,4]] = [
#                     {ele:[1,2,3,5], dist:1},
#                     {ele:[2,2,2,3], dist:3}
#                   ]

# root[[4,3,2,1]] = [
#                     {ele:[4,3,2,2], dist:1},
#                     {ele:[4,3,2,3], dist:2}
#                   ]

# root[[2,2,3,9]] = []