from PIL import Image, ImageChops
import PIL.ImageOps 
import os, sys
import numpy

def get_files(path):
  ''' Retorna uma tupla com (caminho_da_imagem, pasta)
  '''

  files_tuple = []
  for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
      if not filename.endswith('_msk.jpg') and filename.endswith('.jpg'):
        path, folder = os.path.split(dirpath)
        folder = folder.replace('/','')
        files_tuple.append((os.path.join(dirpath, filename), folder, filename))

  return files_tuple

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

path_dataset = sys.argv[1]
path_result = sys.argv[2]

images_info = get_files(path_dataset)
print "Images to process: ", len(images_info)
for img_info in images_info:
  img_path = img_info[0]
  mask_path = img_path.replace(".jpg", "_msk.jpg")
  img_folder = img_info[1]
  filename = img_info[2]
  save_path_folder = path_result + "/" + img_folder + "/"
  img_save_path = save_path_folder + filename
  img = Image.open(img_path)
  img = img.convert('L')
  
  mask = Image.open(mask_path)
  mask = PIL.ImageOps.invert(mask)
  mask = mask.convert('L')
  mask = numpy.array(mask)
  img = numpy.array(img)
  mask[mask > 0] = 255
  img[mask==0] = 0
  img_res = Image.fromarray(img)
  img_res = trim(img_res)
  
  if not os.path.exists(save_path_folder):
    os.makedirs(save_path_folder)

  img_res.save(img_save_path)

  
  #res_img.show()
  #
  #im.show()