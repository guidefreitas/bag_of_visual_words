from collections import namedtuple
import argparse
import os
from random import randrange
import shutil

def get_files(path):
  ''' Retorna uma tupla com (caminho_da_imagem, pasta)
  '''

  files_tuple = []
  for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
      if filename.lower().endswith('jpg') or filename.lower().endswith('png') or filename.lower().endswith('ppm'):
        path, folder = os.path.split(dirpath)
        folder = folder.replace('/','')
        files_tuple.append((os.path.join(dirpath, filename), folder, filename))

  return files_tuple

def create_folder(folder_path):
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)

def split(dir_images, dir_train, dir_test, split_perc):
  images_info = get_files(dir_images)
  categories = {}
  for image_info in images_info:
    image_path = image_info[0]
    category = image_info[1]
    filename = image_info[2]
    if not category in categories:
      categories[category] = []    
    categories[category].append((image_path, category, filename))

  for category in categories:
    print "Splitting category: " + str(category)
    images_info = categories[category]
    train_images_info = []
    test_images_info = []
    num_img_test = int(len(images_info) * split_perc)
    path_category_train = os.path.join(dir_train, category)
    path_category_test = os.path.join(dir_test, category)
    create_folder(path_category_train)
    create_folder(path_category_test)
    for i in range(num_img_test):
      random_index = randrange(0,len(images_info))
      image_info = images_info[random_index]
      test_images_info.append(image_info)
      images_info.remove(image_info)
    train_images_info = images_info

    print "Saving train images"
    for image_info in train_images_info:
      image_path = image_info[0]
      category_image = image_info[1]
      filename = image_info[2]
      dest_path = os.path.join(dir_train,category,filename)
      shutil.copy2(image_path, dest_path)

    print "Saving test images"
    for image_info in test_images_info:
      image_path = image_info[0]
      category_image = image_info[1]
      filename = image_info[2]
      dest_path = os.path.join(dir_test,category,filename)
      shutil.copy2(image_path, dest_path)


def main():
  parser = argparse.ArgumentParser(prog='test_splitter.py', usage='%(prog)s [options]',
      description='Print table result', 
      epilog="")
  parser.add_argument('-dir_images', '--dir_images', help='input directory with files', required=True)
  parser.add_argument('-dir_train', '--dir_train', help='output directory with train files', required=True)
  parser.add_argument('-dir_test', '--dir_test', help='output directory with test files', required=True)
  parser.add_argument('-split', '--split', help='split percentage for test images (0.0 - 0.9)', required=True)
  args = parser.parse_args()

  dir_images = args.dir_images
  dir_train = args.dir_train
  dir_test = args.dir_test
  split_perc = float(args.split)

  split(dir_images, dir_train, dir_test, split_perc)


if __name__ == "__main__":
  main()