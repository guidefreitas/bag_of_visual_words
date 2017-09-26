from pylearn2.datasets.stl10 import STL10
import numpy

train = STL10(which_set = 'train', center = False)
test = STL10(which_set = 'test', center = False)
print dir(train)
class_names = train.class_names
train_X = numpy.cast['float32'](train.get_data()[0])
assert train_X.shape == (8000, 96*96*3)
train_y = train.get_data()[1]
assert train_y.shape == (8000,)

print "#############################################################"
print "X shape: ", train_X.shape
print "#############################################################"

