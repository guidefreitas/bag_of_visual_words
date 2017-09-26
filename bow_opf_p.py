from __future__ import division
import cv2
from PIL import Image
import numpy
#import vlfeat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import filesprocess
import kmeans
import featureextractor
import time
import sys, os
import cPickle as pickle
from subprocess import call
from sklearn import preprocessing
from sklearn.cluster import KMeans
import libopf_py
from sklearn import metrics
from random import randrange
import multiprocessing
import itertools

def extract_descriptor_one(image_info, self):
  img_path = image_info[0]
  img_folder = image_info[1]
  features, descriptors = featureextractor.extract_descriptor(img_path, self.thumbnail_size, self.feature_type, self.descriptor_type)
  tmp_desc = []
  if descriptors != None:
    #self.log("Descriptors lenght: " + str(len(descriptors)))
    for desc in descriptors:
      tmp_desc.append(desc)
  return tmp_desc

def func_star_extract_descriptor_one(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return extract_descriptor_one(*a_b)

def extract_descriptors_and_predict(image_info, self):
  img_path = image_info[0]
  img_folder = image_info[1]
  self.log("Processing: " + str(img_path))
  t_ext_desc_start = time.time()
  features, descriptors = featureextractor.extract_descriptor(img_path, self.thumbnail_size,self.feature_type,self.descriptor_type)
  t_ext_desc_end = time.time() - t_ext_desc_start

  result_classes = []
  result_data = []
  t_cluster_consult_start = time.time()
  if (descriptors != None):
    labels = self.kmeans_cls.predict(descriptors)        
    bins = range(self.kmeans_k)
    labels_hist = numpy.histogram(labels, bins=bins, density=True)[0]
    result_classes.append(img_folder)
    result_data.append(labels_hist)
  t_cluster_consult_end = time.time() - t_cluster_consult_start
  self.log("Time cluster consult: " + str(t_cluster_consult_end))
  return (result_classes, result_data, t_ext_desc_end, t_cluster_consult_end)

def func_star_extract_descriptors_and_predict(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return extract_descriptors_and_predict(*a_b)

def extract_descriptors_predict_test(image_info, params):
  self = params[0]
  le = params[1]
  img_path = image_info[0]
  img_folder = image_info[1]
  self.log("Processing (desc+predict test): " + str(img_path))
  label_test = le.transform([img_folder])
  labels_hist_array = []
  labels_test = []
  features, descriptors = featureextractor.extract_descriptor(img_path, self.thumbnail_size,self.feature_type,self.descriptor_type)
  if (descriptors != None):
    labels = self.kmeans_cls.predict(descriptors)
    bins = range(self.kmeans_k)
    labels_hist = numpy.histogram(labels, bins=bins, density=True)[0]
    labels_hist_array.append(labels_hist)
    labels_test.append(label_test)
  return (labels_hist_array, labels_test)

def func_star_extract_descriptors_predict_test(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return extract_descriptors_predict_test(*a_b)

class BoWOpfP:
  train_path = ""
  test_path = ""
  results_path = ""
  kmeans_k = 100
  thumbnail_size = (50,50)
  kmeans_cls = None
  svm_cls = None
  verbose = False
  result_clustering = None
  result_predict = None
  result_classes = None
  result_data = None
  descriptor_type = None
  feature_type = None
  n_sample_images = 500
  n_sample_descriptors = 2000
  params_output = {}
  distance_function = "euclidian" #0.419           
  # distance_function = "log_euclidian" #0.431     
  # distance_function = "chi_square" #0.396         
  # distance_function = "manhattan" #0.435         
  # distance_function = "canberra" #0.456          
  # distance_function = "squared_chord"      
  # distance_function = "squared_chi_square" #0.460
  # distance_function = "bray_curtis" #0.437 

  def log(self, msg):
    if self.verbose:
      print msg

  def print_params(self):
    print "train_path: ", self.train_path
    print "test_path: ", self.test_path
    print "results_path: ", self.results_path
    print "kmeans_k: ", self.kmeans_k
    print "thumbnail_size: ", self.thumbnail_size

  def __init__(self, train_path, 
    test_path, results_path, 
    n_sample_images=500,
    n_sample_descriptors=2000,
    kmeans_k=100, 
    thumbnail_size = (50,50), 
    descriptor_type=None, feature_type=None,
    verbose=False):

    if descriptor_type == None or feature_type == None:
      raise Exception("Descriptor type or Feature type cannot be None")
      
    self.train_path = train_path
    self.test_path = test_path
    self.n_sample_images = n_sample_images
    self.n_sample_descriptors = n_sample_descriptors
    self.kmeans_k = kmeans_k
    self.thumbnail_size = thumbnail_size
    self.verbose = verbose
    self.descriptor_type = descriptor_type
    self.feature_type = feature_type

    self.params_output['n_sample_images'] = self.n_sample_images
    self.params_output['n_sample_descriptors'] = self.n_sample_descriptors
    self.params_output['k_value'] = self.kmeans_k
    self.params_output['thumbnail_size'] = self.thumbnail_size
    self.params_output['feature_type'] = self.feature_type
    self.params_output['descriptor_type'] = self.descriptor_type

    if self.verbose:
      self.print_params()


  def train(self):
    self.log("BoW - Starting Train")
    self.log("train_path: " + self.train_path)
    self.kmeans_cls = KMeans(init='k-means++', n_clusters=self.kmeans_k, n_init=10, n_jobs=-1)
    result_descriptors = []
    self.log("Listing files to process")
    total_images_info = filesprocess.get_files(self.train_path)
    images_info = []
    if len(total_images_info) > self.n_sample_images:
      for i in range(self.n_sample_images):
        random_index = randrange(0,len(total_images_info))
        images_info.append(total_images_info[random_index])
    else:
      images_info = total_images_info
      
    self.log(str(len(images_info)) + " files to process.")

    pool = multiprocessing.Pool()
    tmp_desc = []
    descs_data = pool.map(func_star_extract_descriptor_one, 
                          itertools.izip(images_info, 
                          itertools.repeat(self)))

    for data in descs_data:
      tmp_desc.extend(data)
    
    self.log("Desc lenght original: " + str(len(tmp_desc)))
    tmp_desc = numpy.array(tmp_desc)
    if len(tmp_desc) > self.n_sample_descriptors:
      rand = numpy.random.permutation(self.n_sample_descriptors)
      tmp_desc = tmp_desc[rand]
    self.log("Desc lenght reduced: " + str(len(tmp_desc)))
    self.params_output['real_desc_size'] = len(tmp_desc)
    self.log("Kmeans fit")
    t_kmeans_start = time.time()
    self.kmeans_cls.fit(tmp_desc)
    t_kmeans_end = time.time() - t_kmeans_start
    self.params_output['time_cluster_fit'] = t_kmeans_end
    self.log("Time cluster fit: " + str(t_kmeans_end))

    self.log("Generating histograms")
    self.result_classes = []
    self.result_data = [] 
    descs_data = []
    #(result_classes, result_data, t_ext_desc_end, t_cluster_consult_end)
    descs_data = pool.map(func_star_extract_descriptors_and_predict, 
                          itertools.izip(total_images_info, 
                          itertools.repeat(self)))
    
    t_ext_desc_sum = 0.0
    t_cluster_consult_sum = 0.0
    for data in descs_data:
      self.result_classes.extend(data[0])
      self.result_data.extend(data[1])
      t_ext_desc_sum = t_ext_desc_sum + data[2]
      t_cluster_consult_sum = t_cluster_consult_sum + data[3]

    pool.close()
    pool.terminate()

    self.params_output['time_ext_desc_med'] =  t_ext_desc_sum / len(total_images_info)
    self.params_output['time_cons_cluster_med'] =  t_cluster_consult_sum / len(total_images_info)

  def save_pickle(self, filename):
    output = open(filename, 'wb')
    data = (self.svm_cls, self.kmeans_cls)
    pickle.dump(data, output)
    output.close()

  def load_pickle(self, filename):
    input_file = open(filename, 'rb')
    data = pickle.load(input_file)
    self.svm_cls = data[0]
    self.kmeans_cls = data[1]

  def run_opf_supervised(self):

    le = preprocessing.LabelEncoder()
    le.fit(self.result_classes)
    #list(le.classes_)
    cross_result_classes = le.transform(self.result_classes)
    cross_result_classes = cross_result_classes.astype(numpy.int32)
    
    O = libopf_py.OPF()
    result_data_array = numpy.array(self.result_data)
    self.log("Training OPF")
    t_opf_start = time.time()
    O.fit(result_data_array, cross_result_classes,metric=self.distance_function)
    t_opf_end = time.time() - t_opf_start
    self.params_output['time_classificator_fit'] = t_opf_end
    images_info = filesprocess.get_files(self.test_path)

    labels_test = []
    labels_hist_array = []

    pool = multiprocessing.Pool()
    preds_data = []
    preds_data = pool.map(func_star_extract_descriptors_predict_test, 
                          itertools.izip(images_info, 
                          itertools.repeat((self, le))))

    for data in preds_data:
      labels_hist_array.extend(data[0])
      labels_test.extend(data[1])

    pool.close()
    pool.terminate()

    labels_hist_array = numpy.asarray(labels_hist_array, numpy.float64)
    labels_predicted = O.predict(labels_hist_array)

    accuracy = metrics.accuracy_score(labels_test, labels_predicted)
    precision = metrics.precision_score(labels_test, labels_predicted)
    recall = metrics.recall_score(labels_test, labels_predicted)
    f1 = metrics.f1_score(labels_test, labels_predicted)

    self.params_output['accuracy'] = accuracy
    self.params_output['precision'] = precision
    self.params_output['recall'] = recall
    self.params_output['F1'] = f1

    self.log("Accuracy: " + str(accuracy))
    self.log("Precision: " + str(precision))
    self.log("Recall: " + str(recall))
    self.log("F1: " + str(f1))
    return accuracy, precision, recall, f1


  def process(self):
    self.log("BoW - Starting Process")
    self.train()


