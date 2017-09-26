from PIL import Image, ImageChops, ImageOps 
from sklearn.feature_extraction import image
import os, sys
import numpy
#from skimage.util.shape import view_as_blocks
import argparse

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


def resize1(img, img_size):
  if img.size[0] >= img.size[1]:
    wpercent = (img_size/float(img.size[1]))
    new_width = int((float(img.size[0])*float(wpercent)))
    img = img.resize((new_width, img_size))
  elif img.size[1] >= img.size[0]:
    wpercent = (img_size/float(img.size[0]))
    new_height = int((float(img.size[1])*float(wpercent)))
    img = img.resize((img_size, new_height))
  return img

def extract_patches(img, thumb_size, img_size, max):
  img_patches = []
  
  if args.verbose:
    print "Creating thumbnail with size ", thumb_size 

  img = resize1(img, thumb_size)
  #img.thumbnail((thumb_size, thumb_size))
  print img.size
  if img.size[0] > img_size or img.size[1] > img_size:
    img_array = numpy.array(img)
    try:
      patches = image.extract_patches_2d(image=img_array, patch_size=(img_size, img_size), max_patches=max, random_state=0)
    except:
      patches = image.extract_patches_2d(image=img_array, patch_size=(img_size, img_size), max_patches=1, random_state=0)
    for patch in patches:
      img_patches.append(patch)
    
  else:
    img_array = numpy.array(img)
    img_patches.append(img_array)
  
  if args.verbose:
    print "Patches extracted: ", len(img_patches)
  return img_patches

if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog='python square_images.py', usage='%(prog)s [options]',
    description='Convert images to the size specifield in -S option', 
    epilog="")
  parser.add_argument('-s', '--source', help='source directory', required=True)
  parser.add_argument('-d', '--destination', help='destination directory', required=True)
  parser.add_argument('-S', '--size', default=231, type=int,  help='new image size (default is 32) (e.g. 32)', required=True)
  parser.add_argument('-f', '--flip', default='false', choices=['true','false'], help='add horizontal flipped images (true or false)')
  parser.add_argument('-m', '--mode', default='distorce', choices=['distorce','crop'], help='convertion mode to crop or distorce image modes - default is distorce')
  parser.add_argument('-t', '--thumb_size', default=256, type=int ,help='thumbnail size to be used when mode crop is selected')

  parser.add_argument('-v', default=False, type=bool, dest='verbose')
  args = parser.parse_args()
  
  src_dir = args.source
  dst_dir = args.destination
  img_size = int(args.size)
  do_flip = args.flip
  conv_mode = args.mode

  print "Source: ", src_dir
  print "Destiny: ", dst_dir
  print "Image size: ", img_size

  img_to_process = get_files(src_dir)
  for idx, img_info in enumerate(img_to_process):
    print "Processing ", idx, " of ", len(img_to_process), ": ", img_info[0]
    img = Image.open(img_info[0])
    img.load()
    img = img.convert("RGB")
    
    if conv_mode == 'crop':
      img_patches = extract_patches(img, args.thumb_size, img_size, 5)
    elif conv_mode == 'distorce':
      img_patches = []
      img = img.resize((img_size, img_size), Image.ANTIALIAS)
      img_array = numpy.array(img)
      img_patches.append(img_array)
    else:
      raise Exception('convertion mode unspecified')

    folder_path = dst_dir + "/" + img_info[1]
    if not os.path.exists(folder_path):
      os.makedirs(folder_path)

    for idx_patch, img_patch in enumerate(img_patches):
      if img_patch.shape[0] == img_size and img_patch.shape[1] == img_size: 
        if do_flip == 'true':
          file_save_path = folder_path + "/" + str(idx_patch) + "_" + img_info[2].lower()
          file_save_path_flipped = folder_path + "/" + str(idx_patch) + "_flipped_" + img_info[2].lower()
          img = Image.fromarray(img_patch)
          img.save(file_save_path)
          img = ImageOps.mirror(img)
          img.save(file_save_path_flipped.replace(".png",".jpg"), 'JPEG', quality=100)
        else:
          file_save_path = folder_path + "/" + str(idx_patch) + "_" + img_info[2].lower()
          img = Image.fromarray(img_patch)
          img.save(file_save_path.replace(".png",".jpg"), 'JPEG', quality=100)


