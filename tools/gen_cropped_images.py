import lsvrc
import cv2

path_to_process = '/Users/guilherme/Downloads/ILSVC'
base_cropped_path = '/Volumes/Dados/Downloads/ILSVRC2013/val_cropped'
files_processed = lsvrc.process_directory(path_to_process, base_cropped_path)

print str(len(files_processed)), " files processed."