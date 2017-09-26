import lsvrc
import cv2

#path_to_process = '/Users/guilherme/Downloads/ILSVRC'
base_desc_path = '/Volumes/Dados/Downloads/ILSVRC2013/desc_files'
#files_processed = lsvrc.generate_descriptors_file(path_to_process, base_desc_path, 200)
#print str(len(files_processed)), " files processed."

files_desc = lsvrc.get_desc_files_list(base_desc_path)
for file_desc in files_desc:
  print "Processing: ", file_desc
  files_info = lsvrc.load_desc_file(file_desc)
  for file_info in files_info:
    print  "File: ", file_info[0], len(file_info[3])