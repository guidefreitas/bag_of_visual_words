from sklearn.base import BaseEstimator
import os, sys
from subprocess import call
import numpy

class Opf(BaseEstimator):

  def _remove_temp_files(self, filename):
    tmp_files = [filename, filename.replace(".txt",".dat"), "testing.dat", "testing.dat.acc", "testing.dat.time", "training.dat", "training.dat.time", "training.dat.out"]
    for tmp_file in tmp_files:
      if os.path.isfile(tmp_file):
        os.remove(tmp_file)

  def _create_file_train(self, X, y):
    opf_filename = 'opf_data.txt'
    self._remove_temp_files(opf_filename)
    num_samples = X.shape[0]
    num_features = X.shape[1]

    unique_classes = numpy.asarray(y)
    unique_classes = numpy.unique(unique_classes)
    unique_classes = numpy.sort(unique_classes)
    num_classes = len(unique_classes)

    cross_result_classes = []
    for clazz in y:
      idx = numpy.where(unique_classes == clazz)[0][0]
      cross_result_classes.append(int(idx))

    cross_result_classes = numpy.array(cross_result_classes, dtype=numpy.uint8)

    opf_file = open(opf_filename, 'wb')
    opf_file.write(str(num_samples) + " " + str(num_classes) + " " + str(num_features) + "\n")
    
    for idx, sample in enumerate(cross_result_classes):
      sample_idx = idx
      sample_class = cross_result_classes[idx]
      sample_hist = X[idx]
      sample_data = str(sample_idx) + " " + str(sample_class) + " "
      for h in sample_hist:
        sample_data = sample_data + str(h) + " "
      sample_data = sample_data + "\n"
      opf_file.write(sample_data)

    opf_file.close()

  def _create_file_predict(self, X, opf_filename):
    opf_file = open(opf_filename, 'wb')
    if len(X.shape) == 1:
      num_samples = 1
      num_features = X.shape[0]
      predict_file = open(opf_filename, "wb")
      predict_file.write(str(num_samples) + " 1 " + str(num_features) + "\n")
      for idx_data, data in enumerate(X):
        line_data = str(idx_data) + " 0" + " " + str(data)
        predict_file.write(line_data)
      predict_file.close()
    else:
      num_samples = X.shape[0]
      num_features = X.shape[1]

      predict_file = open(opf_filename, "wb")
      predict_file.write(str(num_samples) + " 1 " + str(num_features) + "\n")
      for idx_data, data in enumerate(X):
        line_data = str(idx_data) + " 0"
        for d in data:
          line_data = line_data + " " + str(d)
        line_data = line_data + "\n"
        predict_file.write(line_data)
      predict_file.close()

  def fit(self, X, y=None):
    self._create_file_train(X, y)
    opf_filename = 'opf_data.txt'
    print "Calling txt2opf"
    call(["LibOPF/tools/txt2opf", opf_filename, opf_filename.replace(".txt", ".dat")])
    call(["LibOPF/bin/opf_train", opf_filename.replace(".txt", ".dat")])


  def predict(self, X):
    opf_filename_predict = 'opf_data_predict.txt'
    self._create_file_predict(X, opf_filename_predict)
    call(["LibOPF/tools/txt2opf", opf_filename_predict, opf_filename_predict.replace(".txt", ".dat")])
    call(["LibOPF/bin/opf_classify", opf_filename_predict.replace(".txt", ".dat")])

    predict_result = []
    opf_clustet_out_file = open(opf_filename_predict.replace(".txt", ".dat.out"), "rb")
    for line in opf_clustet_out_file:
      if line != "":
        predict_result.append(int(line))
    opf_clustet_out_file.close()
    return predict_result

  def accuracy(self, X):
    opf_filename_accuracy = 'opf_data_accuracy.txt'
    self._create_file_predict(X, opf_filename_accuracy)
    call(["LibOPF/tools/txt2opf", opf_filename_accuracy, opf_filename_accuracy.replace(".txt", ".dat")])
    call(["LibOPF/bin/opf_classify", opf_filename_accuracy.replace(".txt", ".dat")])
    call(["LibOPF/bin/opf_accuracy", opf_filename_accuracy.replace(".txt", ".dat")])
    opf_acc = open(opf_filename_accuracy.replace(".txt", ".dat.acc"), "rb")
    accuracy = float(opf_acc.readline())
    opf_acc.close()
    opf_time_file = open(opf_filename_accuracy.replace(".txt", ".dat.time"), "rb")
    opf_time = opf_time_file.readline()
    opf_time_file.close()
    return accuracy, opf_time
