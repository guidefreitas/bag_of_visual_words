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

class BoW:
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
  n_sample_images = 0
  n_sample_descriptors = 2000
  params_output = {}

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

    tmp_desc = []
    for idx, image_info in enumerate(images_info):
      self.log("Processing " + str(idx) + " of " + str(len(images_info)))
      img_path = image_info[0]
      img_folder = image_info[1]
      features, descriptors = featureextractor.extract_descriptor(img_path, self.thumbnail_size,self.feature_type,self.descriptor_type)
      if descriptors != None:
        for desc in descriptors:
          tmp_desc.append(desc)

    self.log("Desc lenght original: " + str(len(tmp_desc)))
    tmp_desc = numpy.array(tmp_desc)
    if len(tmp_desc) > self.n_sample_descriptors:
      rand = numpy.random.permutation(self.n_sample_descriptors)
      tmp_desc = tmp_desc[rand]
    
    self.log("Desc lenght reduced: " + str(len(tmp_desc)))
    self.params_output['real_desc_size'] = len(tmp_desc)
    self.kmeans_cls = KMeans(init='k-means++', n_clusters=self.kmeans_k, n_init=10, n_jobs=-1)
    self.log("Kmeans fit")
    t_kmeans_start = time.time()
    self.kmeans_cls.fit(tmp_desc)
    t_kmeans_end = time.time() - t_kmeans_start
    self.params_output['time_cluster_fit'] = t_kmeans_end
    self.log("Time cluster fit: " + str(t_kmeans_end))

    self.log("Generating histograms")
    result_labels = []
    self.result_classes = []
    self.result_data = []
    self.result_clustering = []
    tmp_desc = []
    t_ext_desc_sum = 0
    t_cluster_consult_sum = 0
    for idx, image_info in enumerate(total_images_info):
      self.log("Generating hist " + str(idx) + " of " + str(len(total_images_info)))
      img_path = image_info[0]
      img_folder = image_info[1]
      t_ext_desc_start = time.time()
      features, descriptors = featureextractor.extract_descriptor(img_path, self.thumbnail_size,self.feature_type,self.descriptor_type)
      t_ext_desc_end = time.time() - t_ext_desc_start
      t_ext_desc_sum = t_ext_desc_sum + t_ext_desc_end
      self.log("Time desc extraction: " + str(t_ext_desc_end))
      t_cluster_consult_start = time.time()
      if (descriptors != None):
        labels = self.kmeans_cls.predict(descriptors)
        bins = range(self.kmeans_k)
        labels_hist = numpy.histogram(labels, bins=bins, density=True)[0]
        self.result_clustering.append((img_path, img_folder, labels_hist))
        self.result_classes.append(img_folder)
        self.result_data.append(labels_hist)
      t_cluster_consult_end = time.time() - t_cluster_consult_start
      t_cluster_consult_sum = t_cluster_consult_sum + t_cluster_consult_end
      self.log("Time cluster consult: " + str(t_cluster_consult_end))

    self.params_output['time_ext_desc_med'] =  t_ext_desc_sum / len(total_images_info)
    self.params_output['time_cons_cluster_med'] =  t_cluster_consult_sum / len(total_images_info)

    self.log("Creating SVM")
    self.svm_cls = svm.LinearSVC(C=1.0, loss='l2', class_weight='auto')
    t_svm_start = time.time()
    self.svm_cls.fit(self.result_data, self.result_classes)
    t_svm_end = time.time() - t_svm_start
    self.params_output['time_classificator_fit'] = t_svm_end

    
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

    
  def run_test(self):
    labels_test = []
    labels_predicted = []

    images_info = filesprocess.get_files(self.test_path)
    for idx, image_info in enumerate(images_info):
      self.log("Generating hist " + str(idx) + " of " + str(len(images_info)))
      img_path = image_info[0]
      img_folder = image_info[1]
      features, descriptors = featureextractor.extract_descriptor(img_path, self.thumbnail_size,self.feature_type,self.descriptor_type)
      if (descriptors != None):
        labels = self.kmeans_cls.predict(descriptors)
        bins = range(self.kmeans_k)
        labels_hist = numpy.histogram(labels, bins=bins, density=True)[0]
        prediction = self.svm_cls.predict(labels_hist)
        confidence = self.svm_cls.decision_function(labels_hist)
        # print "Classes: ", self.svm_cls.classes_
        # print "Prediction: ", prediction[0]
        # print "Confidence: ", confidence[0]
        labels_predicted.append(prediction[0])
        labels_test.append(img_folder)
        #self.log("Prediction: " + str(prediction[0]))
        #self.log("Real: " + str(img_folder))


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

