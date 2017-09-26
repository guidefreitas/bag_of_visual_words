import cPickle as pickle
import cv2
import numpy
from PIL import Image
import zlib
import filesprocess
import featureextractor

n_chunk = 2000
dir_path = '/home/images/CALTECH256/train'
output_dir = ""
zipped = True
files = filesprocess.get_files(dir_path)
files_chunks = zip(*[iter(files)]*n_chunk)
print "Chunks: ", len(files_chunks)
for idx_chunk, chunk in enumerate(files_chunks):
  chunk_data = []
  for idx_file, file_info in enumerate(chunk):
    img_path = file_info[0]
    folder = file_info[1]
    print "Processing: ", idx_file, " - ", img_path
    keypoints, descriptors = featureextractor.extract_descriptor(img_path, (512,512), "SURF", "SURF")
    data = {'path':img_path, 'folder':folder, 'descriptors':descriptors}
    chunk_data.append(data)
  print "Saving chunk ", idx_chunk
  output_filename = output_dir + "caltech256_" + str(idx_chunk) + ".pickle"
  
  if zipped:
    with open(output_filename + ".gz", 'wb') as fp:
     fp.write(zlib.compress(pickle.dumps(chunk_data, pickle.HIGHEST_PROTOCOL),9))
  else:
    with open(output_filename, 'wb') as fp:
      fp.write(pickle.dumps(chunk_data, pickle.HIGHEST_PROTOCOL))
  #output = open(output_filename, 'wb')
  #pickle.dump(chunk_data, output)
  #output.close()
# img_path = '/home/images/CALTECH256/train/010.beer-mug/010_0004.jpg'
# img = Image.open(img_path)
# img.load()
# img_array = numpy.array(img)
# img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
# featureDetector = cv2.FeatureDetector_create("SIFT")
# descriptorDetector = cv2.DescriptorExtractor_create("SIFT")




# data = []
# for i in range(10):
#   keypoints = featureDetector.detect(img_gray)
#   keypoints, descriptors = descriptorDetector.compute(img_gray, keypoints)
#   print i, " Descriptors: ", len(descriptors)
#   data.append({'path':img_path, 'descriptors':descriptors })
# output = open('test.picle', 'wb')
# print "Saving pickle file"
# pickle.dump(data, output)
# output.close()

# print "Saving compressed pickle file"
# with open('test2.pickle.gz', 'wb') as fp:
#   fp.write(zlib.compress(pickle.dumps(data, pickle.HIGHEST_PROTOCOL),9))