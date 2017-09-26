import bow
import sys, os
import time
import numpy

rc = bow.BoW(train_path=None, 
      validation_path=None, 
      test_path=None, 
      results_path=None,
      verbose=False)

params_filename = sys.argv[2] #"params_500_256_SURF_SURF.pkl"
img_predict_path = sys.argv[1] #'/Volumes/Imagens/ETH80-cropped256/train/car3/car3-066-117.png'
rc.load_pickle(params_filename)
pred = rc.predict(img_predict_path)
pred_prob = rc.predict_prob(img_predict_path)
pred_prob = pred_prob[0]
classes = rc.get_svm_classes()
tops = numpy.argsort(pred_prob)[-10:]
tops = tops[::-1]
print "Prediction", pred
print "Argmax", numpy.argmax(pred_prob)
print "Class", classes[numpy.argmax(pred_prob)]
for idx, top in enumerate(tops):
  print str(idx+1) + ": " + classes[top] + " : " + str(pred_prob[top])
