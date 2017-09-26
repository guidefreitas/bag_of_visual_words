import cPickle
import numpy
from scipy.misc import imsave
import sys, os
from PIL import Image

cifar_path = sys.argv[1]
train_dest_path = sys.argv[2] + '/train'
test_dest_path = sys.argv[2] + '/test'

def unpickle(file):  
  fo = open(file, 'rb')
  dict = cPickle.load(fo)
  fo.close()
  return dict

xs = []
ys = []
for j in range(5):
  d = unpickle(cifar_path + '/' + 'data_batch_' + `j+1`)
  x = d['data']
  y = d['labels']
  xs.append(x)
  ys.append(y)

xs_test = []
ys_test = []
d = unpickle(cifar_path + '/' + 'test_batch')
xs_test.append(d['data'])
ys_test.append(d['labels'])

x = numpy.concatenate(xs)
y = numpy.concatenate(ys)

x_test = numpy.concatenate(xs_test)
y_test = numpy.concatenate(ys_test)

x = numpy.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
x_test = numpy.dstack((x_test[:, :1024], x_test[:, 1024:2048], x_test[:, 2048:]))

for idx, data in enumerate(x):
  img_class = y[idx]
  folder_path = train_dest_path + '/' + str(img_class)
  img_data = data.reshape((32,32,3))
  img = Image.fromarray(img_data)
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)

  img.save(folder_path + '/' + str(idx) + '.jpg')

for idx, data in enumerate(x_test):
  img_class = y[idx]
  folder_path = test_dest_path + '/' + str(img_class)
  img_data = data.reshape((32,32,3))
  img = Image.fromarray(img_data)
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)

  img.save(folder_path + '/' + str(idx) + '.jpg')


#for i in range(50):
#  imsave('cifar10_batch_'+`i`+'.png', x[1000*i:1000*(i+1),:])
#imsave('cifar10_batch_'+`50`+'.png', x[50000:51000,:]) # test set

# dump the labels
#L = 'var labels=' + `list(y[:51000])` + ';\n'
#open('cifar10_labels.js', 'w').write(L)

