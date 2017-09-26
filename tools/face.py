import numpy as np
import cv2
import argparse
import os
from sklearn.feature_extraction import image

face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haar/haarcascade_eye.xml')

def get_files(path):
  ''' Retorna uma tupla com (caminho_da_imagem, pasta)
  '''

  files_tuple = []
  for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
      if not filename.lower().endswith('_msk.jpg') and filename.lower().endswith('.jpg') or filename.lower().endswith('.png') :
        path, folder = os.path.split(dirpath)
        folder = folder.replace('/','')
        files_tuple.append((os.path.join(dirpath, filename), folder, filename))

  return files_tuple

if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog='python square_images.py', usage='%(prog)s [options]',
    description='Convert images to the size specifield in -S option', 
    epilog="")
  parser.add_argument('-s', '--source', help='source directory', required=True)
  parser.add_argument('-d', '--destination', help='destination directory', required=True)
  parser.add_argument('-m', '--margin', default=0.25, type=float,  help='margin percent (0.1)', required=True)
  args = parser.parse_args()

  src_dir = args.source
  dst_dir = args.destination
  margin_size = float(args.margin)

  print "Source: ", src_dir
  print "Destiny: ", dst_dir
  print "Margin size: ", margin_size

  img_to_process = get_files(src_dir)
  for idx, img_info in enumerate(img_to_process):
    print "Processing ", idx, " of ", len(img_to_process), ": ", img_info[0]
    img_path = img_info[0]
    img_filename = img_info[2]
    dest_folder_path = dst_dir + "/" + img_info[1]
    if not os.path.exists(dest_folder_path):
      os.makedirs(dest_folder_path)

    print "Image: ", img_path
    img = cv2.imread(img_path)
    print "Image shape:", img.shape
    mwidth = int(img.shape[0] * margin_size)
    mheight = int(img.shape[1] * margin_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print "Faces: ", len(faces)
    for idx_face,(x,y,w,h) in enumerate(faces):
      face_img = img[y-int(mheight/2):(y+h)+int(mheight/2), x-int(mwidth/2):(x+w)+int(mwidth/2)]
      print "Face shape: ", face_img.shape
      if face_img.shape[0] == 0 or face_img.shape[1] == 0:
        continue
      patches = image.extract_patches_2d(image=face_img, patch_size=((face_img.shape[0]-2), (face_img.shape[1]-2)), random_state=0)
      print "Patches: ", len(patches)
      for idx_patch, patch in enumerate(patches):
        print "Saving patch ", idx_patch
        file_save_path = dest_folder_path + "/" + str(idx_face) + "_" + str(idx_patch) + "_" + img_filename
        cv2.imwrite(file_save_path, face_img)
        flipped_face = cv2.flip(face_img,1)
        file_save_path = dest_folder_path + "/" + str(idx_face) + "_" + str(idx_patch) + "_flipped_" + img_filename
        cv2.imwrite(file_save_path, flipped_face)
      

      
