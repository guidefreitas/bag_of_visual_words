from __future__ import division
import cv2
from PIL import Image
import numpy
#import vlfeat
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import filesprocess
import featureextractor
import time
import sys, os
import cPickle as pickle
from subprocess import call
from sklearn import metrics
from random import randrange
import libopf_py
from sklearn import preprocessing
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift, estimate_bandwidth
import subprocess
from sklearn import svm
import multiprocessing
import itertools

#from sklearn.cluster import DBSCAN
#import warnings

#warnings.filterwarnings('error')

def extract_descriptor_one(image_info, self):
  img_path = image_info[0]
  img_folder = image_info[1]
  self.log("Processing: " + str(img_path))
  features, descriptors = featureextractor.extract_descriptor(img_path, self.thumbnail_size, self.feature_type, self.descriptor_type)
  labels_true = []
  tmp_desc = []
  if descriptors != None:
    #self.log("Descriptors lenght: " + str(len(descriptors)))
    for desc in descriptors:
      labels_true.append(img_folder)
      tmp_desc.append(desc)
  return (tmp_desc, labels_true)

def func_star_extract_descriptor_one(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return extract_descriptor_one(*a_b)

def extract_descriptors_and_predict(image_info, self):
  img_path = image_info[0]
  img_folder = image_info[1]
  self.log("Processing (desc+predict): " + str(img_path))
  result_classes = []
  result_data = []
  t_ext_desc_start = time.time()
  features, descriptors = featureextractor.extract_descriptor(img_path, self.thumbnail_size,self.feature_type,self.descriptor_type)
  t_ext_desc_end = time.time() - t_ext_desc_start
  #self.log("Time desc extraction: " + str(t_ext_desc_end))
  t_cluster_consult_start = time.time()
  if (descriptors != None) and (len(descriptors) > 0):
    label_hist = self.opf_predict(descriptors, self.n_clusters)
    result_classes.append(img_folder)
    result_data.append(label_hist)
  t_cluster_consult_end = time.time() - t_cluster_consult_start
  #self.log("Time cluster consult: " + str(t_cluster_consult_end))
  return (result_classes, result_data, t_ext_desc_end, t_cluster_consult_end)

def func_star_extract_descriptors_and_predict(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return extract_descriptors_and_predict(*a_b)

def extract_descriptors_predict_test(image_info, self):
  img_path = image_info[0]
  img_folder = image_info[1]
  labels_hist_array = []
  labels_test = []
  self.log("Processing (desc+predict test): " + str(img_path))
  features, descriptors = featureextractor.extract_descriptor(img_path, self.thumbnail_size,self.feature_type,self.descriptor_type)
  if (descriptors != None):
    label_hist = self.opf_predict(descriptors, self.n_clusters)
    labels_hist_array.append(label_hist)
    labels_test.append(img_folder)
  return (labels_hist_array, labels_test)

def func_star_extract_descriptors_predict_test(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return extract_descriptors_predict_test(*a_b)

class BoWOpfSvmUnsupP:
  train_path = ""
  test_path = ""
  results_path = ""
  thumbnail_size = (50,50)
  verbose = False
  result_clustering = None
  result_predict = None
  result_classes = None
  result_data = None
  descriptor_type = None
  feature_type = None
  svm_cls = None
  n_clusters = 0
  best_k = 0
  kmax = 0
  unique_predictions = None
  max_prediction = 0
  n_sample_images = 500
  n_sample_descriptors = 2000
  params_output = {}
  # distance_function = "euclidian" #0.419           
  # distance_function = "log_euclidian" #0.431     
  # distance_function = "chi_square" #0.396         
  # distance_function = "manhattan" #0.435         
  # distance_function = "canberra" #0.456          
  # distance_function = "squared_chord"      
  # distance_function = "squared_chi_square" #0.460
  # distance_function = "bray_curtis" #0.437 
  distance_type = "0"
  distance_param = "0.05"

  def log(self, msg):
    if self.verbose:
      print msg

  def print_params(self):
    print "train_path: ", self.train_path
    print "test_path: ", self.test_path
    print "results_path: ", self.results_path
    print "thumbnail_size: ", self.thumbnail_size

  def __init__(self, train_path, 
    test_path, results_path, 
    n_sample_images=500,
    n_sample_descriptors=2000,
    kmax=100,
    thumbnail_size = (50,50), 
    descriptor_type=None, feature_type=None,
    distance_type="0",
    distance_param="0.05",
    verbose=False):

    if descriptor_type == None or feature_type == None:
      raise Exception("Descriptor type or Feature type cannot be None")
      
    self.train_path = train_path
    self.test_path = test_path
    self.n_sample_images = n_sample_images
    self.n_sample_descriptors = n_sample_descriptors
    self.kmax = kmax
    self.thumbnail_size = thumbnail_size
    self.verbose = verbose
    self.descriptor_type = descriptor_type
    self.feature_type = feature_type
    self.distance_type=distance_type
    self.distance_param=distance_param

    self.params_output['n_sample_images'] = self.n_sample_images
    self.params_output['n_sample_descriptors'] = self.n_sample_descriptors
    self.params_output['k_value'] = self.kmax
    self.params_output['thumbnail_size'] = self.thumbnail_size
    self.params_output['feature_type'] = self.feature_type
    self.params_output['descriptor_type'] = self.descriptor_type
    self.params_output['distance_type'] = self.distance_type
    self.params_output['distance_param'] = self.distance_param

    if self.verbose:
      self.print_params()


  def opf_cluster(self, descriptors):
    opf_desc_filename = "tmp/desc_opf.txt"
    opf_desc_filename = os.path.realpath(opf_desc_filename)
    n_samples = len(descriptors)
    n_features = len(descriptors[0])
    result_file = open(opf_desc_filename, "w")
    result_file.write(str(n_samples) + " " + "0" + " " + str(n_features) + "\n")
    for idx, dsc in enumerate(descriptors):
      result_file.write(str(idx) + " ")
      for d in dsc:
        result_file.write(str(d) + " ")
      result_file.write("\n")
    result_file.close()

    opf_c_type = "0"

    proc = subprocess.Popen(["LibOPF/LibOPF/tools/txt2opf", opf_desc_filename, opf_desc_filename.replace(".txt", ".dat")], stdout=subprocess.PIPE)
    out = proc.communicate()[0]
    proc.wait()
    #print out
    t_opf_cluster_start = time.time()
    proc = subprocess.Popen(["LibOPF/LibOPF/bin/opf_cluster", opf_desc_filename.replace(".txt", ".dat"), str(self.kmax), self.distance_type, self.distance_param,], stdout=subprocess.PIPE)
    out = proc.communicate()[0]
    proc.wait()
    self.log(out)
    output = out.split("\n")
    for idx, line in enumerate(output):
      self.log(str(idx) + " - " + str(line))
    if self.distance_type == "1" or self.distance_type == "0":
      self.best_k = int(output[10].replace("Reading data file ...best k ", ""))
      self.n_clusters = int(output[15].replace("Clustering by OPF num of clusters ", ""))
    else:
      self.best_k = int(output[10].replace("Reading data file ...best k ", ""))
      self.n_clusters = int(output[12].replace("Clustering by OPF num of clusters ", ""))
    self.log("################ OPF CLUSTER ###################")
    self.log("Best k:" + str(self.best_k))
    self.log("Clusters: " + str(self.n_clusters))
    self.params_output['real_k_value'] = self.n_clusters
    self.log("###############################################")
    t_opf_cluster_end = time.time() - t_opf_cluster_start
    self.params_output['time_cluster_fit'] = t_opf_cluster_end
    self.log("Time cluster fit: " + str(t_opf_cluster_end))
    return self.best_k, self.n_clusters

  def opf_predict(self, descriptors, n_clusters):
    opf_desc_filename = "tmp/desc_opf_" + str(multiprocessing.current_process().pid) + ".txt"
    opf_desc_filename = os.path.realpath(opf_desc_filename)
    n_samples = len(descriptors)
    n_features = len(descriptors[0])
    result_file = open(opf_desc_filename, "w")
    result_file.write(str(n_samples) + " " + "0" + " " + str(n_features) + "\n")
    for idx, dsc in enumerate(descriptors):
      result_file.write(str(idx) + " ")
      for d in dsc:
        result_file.write(str(d) + " ")
      result_file.write("\n")
    result_file.close()
    proc = subprocess.Popen(["LibOPF/LibOPF/tools/txt2opf", opf_desc_filename, opf_desc_filename.replace(".txt", ".dat")], stdout=subprocess.PIPE)
    out = proc.communicate()[0]
    proc.wait()
    proc = subprocess.Popen(["LibOPF/LibOPF/bin/opf_knn_classify", opf_desc_filename.replace(".txt", ".dat")], stdout=subprocess.PIPE)
    out = proc.communicate()[0]
    proc.wait()
    result_opf_knn = open(opf_desc_filename.replace(".txt", ".dat.out"), 'r')
    result_predict = []
    for line in result_opf_knn:
      if line != None and line != "":
        result_predict.append(float(line))
    #self.log("Predictions: " + str(len(result_predict)))
    labels_hist = numpy.histogram(result_predict, bins=n_clusters, density=True)[0]
    return labels_hist

  def train(self):
    self.log("BoW Opf-Opf - Starting Train")
    self.log("train_path: " + self.train_path)
    result_descriptors = []
    self.log("Listing files to process")
    total_images_info = filesprocess.get_files(self.train_path)

    self.log("Total images: " + str(len(total_images_info)))
    images_info = []
    if len(total_images_info) > self.n_sample_images:
      for i in range(self.n_sample_images):
        random_index = randrange(0,len(total_images_info))
        images_info.append(total_images_info[random_index])
    else:
      images_info = total_images_info

    self.log(str(len(images_info)) + " files to process.")


    #Extrai descritores
    pool = multiprocessing.Pool()
    tmp_desc = []
    labels_true = []
    descs_data = pool.map(func_star_extract_descriptor_one, 
                          itertools.izip(images_info, 
                          itertools.repeat(self)))

    #itertools.repeat((self.thumbnail_size, self.feature_type, self.descriptor_type))))
    for data in descs_data:
      tmp_desc.extend(data[0])
      labels_true.extend(data[1])

    tmp_desc = numpy.array(tmp_desc, numpy.float64)
    labels_true = numpy.asarray(labels_true)
    #Seleciona aleatoriamente um numero de descritores (n_sample_descriptors)
    self.log("Desc lenght original: " + str(len(tmp_desc)))
    if len(tmp_desc) > self.n_sample_descriptors:
      rand = numpy.random.permutation(len(tmp_desc))[0:self.n_sample_descriptors]
      tmp_desc = tmp_desc[rand]
      labels_true = labels_true[rand]
    self.log("Desc lenght reduced: " + str(len(tmp_desc)))
    self.params_output['real_desc_size'] = len(tmp_desc)
    self.log("OPF - Clustering")
    self.best_k, self.n_clusters = self.opf_cluster(tmp_desc)
    

    #Extrai os prototipos com base nos descritores das imagens de treinamento
    self.log("Generating predictions")
    self.result_classes = []
    self.result_data = [] #numpy.zeros((len(total_images_info), self.n_clusters), numpy.float64)
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

    self.params_output['time_ext_desc_med'] =  t_ext_desc_sum / len(total_images_info)
    self.params_output['time_cons_cluster_med'] =  t_cluster_consult_sum / len(total_images_info)
    self.result_data = numpy.asarray(self.result_data, numpy.float64)

    pool.close()
    pool.terminate()

    return self.n_clusters#, self.best_k
 
  def run_opf_supervised(self):
    self.log("Creating SVM")
    self.svm_cls = svm.LinearSVC(C=1.0, loss='l2', class_weight='auto')
    t_svm_start = time.time()
    self.svm_cls.fit(self.result_data, self.result_classes)
    t_svm_end = time.time() - t_svm_start
    self.params_output['time_classificator_fit'] = t_svm_end

    images_info = filesprocess.get_files(self.test_path)
    labels_test = []
    labels_hist_array = []

    pool = multiprocessing.Pool()
    preds_data = []
    preds_data = pool.map(func_star_extract_descriptors_predict_test, 
                          itertools.izip(images_info, 
                          itertools.repeat(self)))

    for data in preds_data:
      labels_hist_array.extend(data[0])
      labels_test.extend(data[1])
        
    labels_hist_array = numpy.asarray(labels_hist_array, numpy.float64)
        
    self.log("Generating predictions")
    labels_predicted = self.svm_cls.predict(labels_hist_array)

    pool.close()
    pool.terminate()
    
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
    n_clusters = self.train()
    return n_clusters


