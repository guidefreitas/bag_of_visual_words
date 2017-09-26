import bow
import sys, os
import time
import numpy
from sklearn.feature_extraction import image
from skimage.util.shape import view_as_blocks
from PIL import Image
import cv2

def extract_blocks(img_gray):
  patch_sizes = [128, 256]
  result = []
  for size in patch_sizes:
    blocks = view_as_blocks(img_gray, (size, size))
    for row in range(blocks.shape[0]):
      for col in range(blocks.shape[1]):
        block = blocks[row, col]
        pred = rc.predict(block)
        if pred == None:
          continue
        pred_prob = rc.predict_prob(block)[0]
        top1 = numpy.argsort(pred_prob)[-1:][0]
        top1_prob = pred_prob[top1]
        tops = numpy.argsort(pred_prob)[-5:]
        tops = tops[::-1]
        result.append((top1_prob, pred[0], row, col, size, block))
        #print "Size", size, "Prediction:", pred, "Argmax:", numpy.argmax(pred_prob), "Class:", classes[numpy.argmax(pred_prob)]
        #for idx, top in enumerate(tops):
        #  print "", idx+1, ": ", classes[top], " : ", pred_prob[top]
        #print "="*80
  return result

def extract_patches(img_gray):
  total_patches = []
  patch_sizes = [96, 128]
  result = []
  for size in patch_sizes:
    patches = image.extract_patches_2d(img_array, (size, size))
    print "Patches", patches.shape
    for idx, patch in enumerate(patches):
      pred = rc.predict(patch)
      if pred == None:
        continue
      pred_prob = rc.predict_prob(patch)[0]
      top1 = numpy.argsort(pred_prob)[-1:][0]
      top1_prob = pred_prob[top1]
      tops = numpy.argsort(pred_prob)[-5:]
      tops = tops[::-1]
      result.append((top1_prob, pred[0], size, idx))
      total_patches.append(patch)
      #print "Size", size, "Prediction:", pred, "Argmax:", numpy.argmax(pred_prob), "Class:", classes[numpy.argmax(pred_prob)]
      #for idx, top in enumerate(tops):
      #  print "", idx+1, ": ", classes[top], " : ", pred_prob[top]
      #print "="*80
  return result, total_patches

rc = bow.BoW(train_path=None, 
      validation_path=None, 
      test_path=None, 
      results_path=None,
      verbose=False)

params_filename = sys.argv[2] #"params_500_256_SURF_SURF.pkl"
img_predict_path = sys.argv[1] #'/Volumes/Imagens/ETH80-cropped256/train/car3/car3-066-117.png'
rc.load_pickle(params_filename)
classes = rc.get_svm_classes()
img = Image.open(img_predict_path)

img_array = numpy.array(img)
img_gray = img_array
if len(img_array.shape) >= 3:
  img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

result, total_patches = extract_patches(img_gray)

result_array = numpy.array(result)
result_sorted = numpy.sort(result_array, axis=0)
result_sorted = result_sorted[::-1]
print "Result", result_sorted[0:5]
top_result = result_sorted[0]
img = Image.fromarray(total_patches[int(top_result[3])])
img.show()