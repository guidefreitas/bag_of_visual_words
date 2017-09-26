import sys, os
import filesprocess
from sklearn.cross_validation import train_test_split
import numpy
import shutil

class Caltech:
  dataset_path = ""
  train_path = ""
  validation_path = ""
  test_path = ""
  train_size=0.5
  validation_size=0.3
  test_size=0.2

  def __init__(self, dataset_path, train_path, validation_path,
    test_path, train_size=0.5, validation_size=0.3, test_size=0.2):
    self.train_size = train_size
    self.validation_size = validation_size
    self.test_size = test_size
    self.dataset_path = dataset_path
    self.train_path = train_path
    self.validation_path = validation_path
    self.test_path = test_path

  def process(self):
    if not os.path.exists(self.train_path):
        os.makedirs(self.train_path)

    if not os.path.exists(self.validation_path):
        os.makedirs(self.validation_path)

    if not os.path.exists(self.test_path):
        os.makedirs(self.test_path)

    images_info = filesprocess.get_files(self.dataset_path)
    a_train, a_test = train_test_split(images_info, test_size=self.test_size, random_state=42)

    print "Saving training data"
    for data in a_train:
      img_folder_path = self.train_path + "/" + data[1]
      img_filename = os.path.split(data[0])[1]
      src_path = data[0]
      dst_path = img_folder_path + "/" + img_filename
      print "Processing train: " + src_path
      if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)
      shutil.copyfile(src_path, dst_path)
    
    print "Saving test data"
    for data in a_test:
      img_folder_path = self.test_path + "/" + data[1]
      img_filename = os.path.split(data[0])[1]
      src_path = data[0]
      dst_path = img_folder_path + "/" + img_filename
      print "Processing test: " + src_path
      if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)
      shutil.copyfile(src_path, dst_path)
    
  

