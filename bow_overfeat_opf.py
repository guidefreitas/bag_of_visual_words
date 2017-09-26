from __future__ import division
import cv2
from PIL import Image
import numpy
#import vlfeat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import filesprocess
#import kmeans
import featureextractor
import time
from sklearn import svm, cross_validation
from sklearn.cluster import KMeans
import sys
import cPickle as pickle
from random import randrange
from sklearn import metrics
import libopf_py
from sklearn import preprocessing

class BoWOverfeatOPF:
  train_path = ""
  test_path = ""
  results_path = ""
  opf_cls = None
  verbose = False
  params_output = {}
  distance_function = "euclidian"          
  # distance_function = "log_euclidian"     
  # distance_function = "chi_square"
  # distance_function = "manhattan"
  # distance_function = "canberra"
  # distance_function = "squared_chord"      
  # distance_function = "squared_chi_square"
  # distance_function = "bray_curtis"

  def log(self, msg):
    if self.verbose:
      print msg

  def print_params(self):
    print "train_path: ", self.train_path
    print "test_path: ", self.test_path
    print "results_path: ", self.results_path
  

  def __init__(self, train_path, 
    test_path, results_path,
    verbose=False):
    self.train_path = train_path
    self.test_path = test_path
    self.verbose = verbose

    self.params_output['n_sample_images'] = ''
    self.params_output['n_sample_descriptors'] = ''
    self.params_output['k_value'] = '4096'
    self.params_output['thumbnail_size'] = '(231, 231)'
    self.params_output['feature_type'] = 'OVERFEAT'
    self.params_output['descriptor_type'] = 'OVERFEAT'
    self.params_output['time_cluster_fit'] = ''

    if self.verbose:
      self.print_params()


  def train(self):
    self.log("BoW Overfeat - Starting Train")
    self.log("train_path: " + self.train_path)
    
    result_descriptors = []
    self.log("Listing files to process")
    total_images_info = filesprocess.get_files(self.train_path)
    #total_images_info = total_images_info[0:100]
    self.result_classes = []
    self.result_data = []
    t_ext_desc_sum = 0
    t_feature_extract_start = time.time()
    for idx, image_info in enumerate(total_images_info):
      self.log("Processing " + str(idx) + " of " + str(len(total_images_info)))
      img_path = image_info[0]
      img_folder = image_info[1]
      t_ext_desc_start = time.time()
      features, descriptors = featureextractor.extract_descriptor(img_path, None,"OVERFEAT","OVERFEAT")
      t_ext_desc_end = time.time() - t_ext_desc_start
      t_ext_desc_sum = t_ext_desc_sum + t_ext_desc_end
      if descriptors != None:
        print "Debug 1: ", descriptors[0].shape
        self.result_data.append(descriptors[0])
        self.result_classes.append(img_folder)
    t_feature_extract_end = time.time() - t_feature_extract_start
    self.params_output['time_ext_desc_med'] =  t_ext_desc_sum / len(total_images_info)
    self.params_output['time_feature_extraction'] = t_feature_extract_end
    
    

  def run_test(self):
    le = preprocessing.LabelEncoder()
    le.fit(self.result_classes)
    #list(le.classes_)
    cross_result_classes = le.transform(self.result_classes)
    cross_result_classes = cross_result_classes.astype(numpy.int32)

    result_data_array = numpy.asarray(self.result_data, numpy.float64)
    self.log("Creating OPF")
    self.opf_cls = libopf_py.OPF()
    t_opf_start = time.time()
    #self.svm_cls.fit(self.result_data, self.result_classes)
    self.opf_cls.fit(result_data_array, cross_result_classes,metric=self.distance_function)
    t_opf_end = time.time() - t_opf_start
    self.params_output['time_classificator_fit'] = t_opf_end

    labels_test = []
    labels_predicted = []

    images_info = filesprocess.get_files(self.test_path)
    for idx, image_info in enumerate(images_info):
      self.log("Generating hist " + str(idx) + " of " + str(len(images_info)))
      img_path = image_info[0]
      img_folder = image_info[1]
      features, descriptors = featureextractor.extract_descriptor(img_path, None,"OVERFEAT","OVERFEAT")
      if (descriptors != None):
        descriptors = numpy.asarray(descriptors, numpy.float64)
        prediction = self.opf_cls.predict(descriptors)
        labels_predicted.append(prediction[0])
        label_trans = le.transform([img_folder])
        labels_test.append(label_trans[0])

    print labels_predicted
    accuracy = metrics.accuracy_score(labels_test, labels_predicted)
    self.log("Accuracy: " + str(accuracy))
    precision = metrics.precision_score(labels_test, labels_predicted)
    self.log("Precision: " + str(precision))
    recall = metrics.recall_score(labels_test, labels_predicted)
    self.log("Recall: " + str(recall))
    f1 = metrics.f1_score(labels_test, labels_predicted)
    self.log("F1: " + str(f1))

    self.params_output['accuracy'] = accuracy
    self.params_output['precision'] = precision
    self.params_output['recall'] = recall
    self.params_output['F1'] = f1
    
    return accuracy, precision, recall, f1

  def process(self):
    self.log("BoW - Starting Process")
    self.train()


  # def run_cross_validation(self, kfold=10):
  #   unique_classes = numpy.asarray(self.result_classes)
  #   unique_classes = numpy.unique(unique_classes)
  #   unique_classes = numpy.sort(unique_classes)

  #   cross_result_classes = []
  #   for clazz in self.result_classes:
  #     idx = numpy.where(unique_classes == clazz)[0][0]
  #     cross_result_classes.append(int(idx))

  #   cross_result_classes = numpy.array(cross_result_classes, dtype=numpy.uint8)
  #   scores = cross_validation.cross_val_score(self.svm_cls, self.result_data, cross_result_classes, cv=kfold)
  #   print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()))
  #   return scores.mean(), scores.std()


  # def predict(self, image):
  #   if type(image) == type(""):
  #     features,descriptors = featureextractor.extract_sift(image, self.thumbnail_size,self.feature_type,self.descriptor_type)  
  #   else:
  #     features,descriptors = featureextractor.extract_descriptor(image, self.thumbnail_size,self.feature_type,self.descriptor_type)
  #   if descriptors == None or len(descriptors) == 0:
  #     return None

  #   labels_hist = self.kmeans_cls.predict_hist(descriptors)
  #   prediction = self.svm_cls.predict(labels_hist)
  #   return prediction

