import sys, os
import time
import argparse
#import bow_opf_unsup
import bow_opf_p
import bow_p
import bow_opf_unsup_p
import bow_opf_svm_unsup_p
import bow_overfeat_svm
import bow_overfeat_opf
import precompute
import multiprocessing

parser = argparse.ArgumentParser(prog='python recog.py', usage='%(prog)s [options]',
    description='Convert images to the size specifield in -S option', 
    epilog="")
parser.add_argument('-i', '--input', default="input.txt", help='input text file to process', required=True)
parser.add_argument('-dir_train', '--dir_train', help='training directory with images', required=True)
parser.add_argument('-dir_test', '--dir_test', help='test directory with images', required=True)
parser.add_argument('-dir_results', '--dir_results', help='directory to save results files', required=True)

args = parser.parse_args()
  

input_file_path = args.input

train_path = args.dir_train 
test_path = args.dir_test
results_path = args.dir_results

print "Processing ", input_file_path
print "Train path:", train_path
print "Test path", test_path
print "Results path:", results_path

input_file = open(input_file_path, 'rb')


def process_overfeat_svm():
  print "Type: OVERFEAT+SVM"
  rc = bow_overfeat_svm.BoWOverfeatSVM(train_path=train_path,
      test_path=test_path, 
      results_path=results_path,
      verbose=True)
  t_start = time.time()
  rc.process()
  accuracy, precision, recall, f1 = rc.run_test()
  t_end = time.time() - t_start
  rc.params_output['time_total'] = t_end
  #result_msg = "OVERFEAT" + "," + str(num_k) + " (IGNORED)" + "," + str(thumbnail) + " (IGNORED)" + "," +  str(feature_type) + "," + str(descriptor_type) + "," + str(accuracy) + "," + str(t_end)
  result_msg = "OVERFEAT+SVM" + "|" + str(rc.params_output)
  print "Result: ", result_msg
  save_result(result_msg, results_path)

def process_overfeat_opf():
  print "Type: OVERFEAT+OPF"
  rc = bow_overfeat_opf.BoWOverfeatOPF(train_path=train_path,
      test_path=test_path, 
      results_path=results_path,
      verbose=True)
  t_start = time.time()
  rc.process()
  accuracy, precision, recall, f1 = rc.run_test()
  t_end = time.time() - t_start
  rc.params_output['time_total'] = t_end
  #result_msg = "OVERFEAT" + "," + str(num_k) + " (IGNORED)" + "," + str(thumbnail) + " (IGNORED)" + "," +  str(feature_type) + "," + str(descriptor_type) + "," + str(accuracy) + "," + str(t_end)
  result_msg = "OVERFEAT+OPF" + "|" + str(rc.params_output)
  print "Result: ", result_msg
  save_result(result_msg, results_path)


def process_kmeans_svm(num_k, n_sample_images, n_sample_descriptors, thumbnail_size, feature_type, descriptor_type):
  print "Type: KMeans +  SVM"
  rc = bow_p.BoWP(train_path=train_path,
      test_path=test_path, 
      results_path=results_path,
      n_sample_images=n_sample_images,
      n_sample_descriptors=n_sample_descriptors,
      kmeans_k=num_k,
      thumbnail_size=thumbnail_size,
      feature_type=feature_type,
      descriptor_type=descriptor_type,
      verbose=True)
  t_start = time.time()
  rc.process()
  #params_filename = "params/params_" + str(num_k) + "_" + str(thumbnail) + "_" + str(feature_type) + "_" + str(descriptor_type) + ".pkl"
  #rc.save_pickle(params_filename)

  accuracy, precision, recall, f1 = rc.run_test()
  t_end = time.time() - t_start
  rc.params_output['time_total'] = t_end
  #result_msg = "BoW (Kmeans + SVM)" + "," + str(num_k) + "," + str(thumbnail) + "," +  str(feature_type) + "," + str(descriptor_type) + "," + str(accuracy) + "," + str(t_end) + "," + str(rc.params_output)
  result_msg = "Kmeans+SVM" + "|" + str(rc.params_output)
  print "Result: ", result_msg
  save_result(result_msg, results_path)

def process_kmeans_opf(num_k, n_sample_images, n_sample_descriptors, thumbnail_size, feature_type, descriptor_type):
  print "Type: KMeans + OPF"
  rc = bow_opf_p.BoWOpfP(train_path=train_path, 
      test_path=test_path, 
      results_path=results_path,
      n_sample_images=n_sample_images,
      n_sample_descriptors=n_sample_descriptors,
      kmeans_k=num_k,
      thumbnail_size=thumbnail_size,
      feature_type=feature_type,
      descriptor_type=descriptor_type,
      verbose=True)
  t_start = time.time()
  rc.process()
  #params_filename = "params/params_" + str(num_k) + "_" + str(thumbnail) + "_" + str(feature_type) + "_" + str(descriptor_type) + ".pkl"
  #rc.save_pickle(params_filename)
  
  accuracy, precision, recall, f1 = rc.run_opf_supervised()
  t_end = time.time() - t_start
  rc.params_output['time_total'] = t_end
  #result_msg = "BoW (Kmeans + OPF)" + "," + str(num_k) + "," + str(thumbnail) + "," +  str(feature_type) + "," + str(descriptor_type) + "," + str(accuracy) + "," + str(t_end) + "," + str(rc.params_output)
  result_msg = "Kmeans+OPF" + "|" + str(rc.params_output)
  print "Result: ", result_msg
  save_result(result_msg, results_path)

def process_opf_opf(num_k, n_sample_images, n_sample_descriptors, thumbnail_size, feature_type, descriptor_type, distance_type, distance_param):
  print "Type: OPF +  OPF"
  rc = bow_opf_unsup_p.BoWOpfUnsupP(train_path=train_path, 
      test_path=test_path, 
      results_path=results_path,
      n_sample_images=n_sample_images,
      n_sample_descriptors=n_sample_descriptors,
      kmax=num_k,
      thumbnail_size=thumbnail_size,
      feature_type=feature_type,
      descriptor_type=descriptor_type,
      distance_type=distance_type,
      distance_param=distance_param,
      verbose=True)
  t_start = time.time()
  n_clusters = rc.process()
  accuracy, precision, recall, f1 = rc.run_opf_supervised()
  t_end = time.time() - t_start
  rc.params_output['time_total'] = t_end
  #result_msg = "BoW (OPF + OPF)" + "," + str(n_clusters) + "," + str(thumbnail) + "," +  str(feature_type) + "," + str(descriptor_type) + "," + str(accuracy) + "," + str(t_end) + "," + str(rc.params_output)
  result_msg = "OPF+OPF" + "|" + str(rc.params_output)
  print "Result: ", result_msg
  save_result(result_msg, results_path)

def process_opf_svm(num_k, n_sample_images, n_sample_descriptors, thumbnail_size, feature_type, descriptor_type,distance_type, distance_param):
  print "Type: OPF +  SVM"
  rc = bow_opf_svm_unsup_p.BoWOpfSvmUnsupP(train_path=train_path, 
      test_path=test_path, 
      results_path=results_path,
      n_sample_images=n_sample_images,
      n_sample_descriptors=n_sample_descriptors,
      kmax=num_k,
      thumbnail_size=thumbnail_size,
      feature_type=feature_type,
      descriptor_type=descriptor_type,
      distance_type=distance_type, 
      distance_param=distance_param,
      verbose=True)
  t_start = time.time()
  n_clusters = rc.process()
  #params_filename = "params/params_" + str(kmeans_k) + "_" + str(thumbnail) + "_" + str(feature_type) + "_" + str(descriptor_type) + ".pkl"
  #rc.save_pickle(params_filename)
  
  accuracy, precision, recall, f1 = rc.run_opf_supervised()
  t_end = time.time() - t_start
  rc.params_output['time_total'] = t_end
  #result_msg = "BoW (OPF + SVM)" + "," + str(n_clusters) + "," + str(thumbnail) + "," +  str(feature_type) + "," + str(descriptor_type) + "," + str(accuracy) + "," + str(t_end)
  result_msg = "OPF+SVM" + "|" + str(rc.params_output)
  
  print "Result: ", result_msg
  save_result(result_msg, results_path)


def save_result(message, results_path):
  if not os.path.exists(results_path):
    os.makedirs(results_path)

  result_file = open(results_path + "/results.txt", "a")
  result_file.write(message + "\n")
  result_file.close()

for line in input_file:
  if line.startswith("#") or line == None:
    continue

  
  data = line.split(",")
  process_type = data[0]
  num_k = int(data[1])
  n_sample_images = int(data[2])
  n_sample_descriptors = int(data[3])
  thumbnail = int(data[4])
  thumbnail_size = (thumbnail, thumbnail)
  feature_type = str(data[5])
  descriptor_type = str(data[6])
  if process_type == "OPF+OPF" or process_type == "OPF+SVM":
    distance_type = str(data[7])
    distance_param = str(data[8])

  print "Num K: ", num_k
  print "Thumbnail size: ", thumbnail_size
  print "Feature type: ", feature_type
  print "Descriptor type: ", descriptor_type
  print "Precomputing descriptors"

  pool = None
  if feature_type == "OVERFEAT" or descriptor_type == "OVERFEAT":
    pool = multiprocessing.Pool(processes=1)
  else:  
    pool = multiprocessing.Pool()
  precompute.precompute(train_path, test_path, feature_type, descriptor_type, thumbnail_size, pool)
  pool.close()
  pool.terminate()
  
  if process_type == "KMeans+SVM":
     process_kmeans_svm(num_k=num_k, n_sample_images=n_sample_images, 
      n_sample_descriptors=n_sample_descriptors, thumbnail_size=thumbnail_size,
      feature_type=feature_type, descriptor_type=descriptor_type)
  elif process_type == "KMeans+OPF":
    process_kmeans_opf(num_k=num_k, n_sample_images=n_sample_images, 
        n_sample_descriptors=n_sample_descriptors, thumbnail_size=thumbnail_size,
        feature_type=feature_type, descriptor_type=descriptor_type)
  elif process_type == "OPF+OPF":
    process_opf_opf(num_k=num_k, n_sample_images=n_sample_images, 
        n_sample_descriptors=n_sample_descriptors, thumbnail_size=thumbnail_size,
        feature_type=feature_type, descriptor_type=descriptor_type,
        distance_type=distance_type, distance_param=distance_param)
  elif process_type == "OPF+SVM":
    process_opf_svm(num_k=num_k, n_sample_images=n_sample_images, 
        n_sample_descriptors=n_sample_descriptors, thumbnail_size=thumbnail_size,
        feature_type=feature_type, descriptor_type=descriptor_type,
        distance_type=distance_type, distance_param=distance_param)
  elif process_type == "OVERFEAT+SVM":
    process_overfeat_svm()
  elif process_type == "OVERFEAT+OPF":
    process_overfeat_opf()
  else:
    print "Invalid process type: ", process_type
input_file.close()

