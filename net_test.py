from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet
import numpy

size_x = 20
hidden = 10
net = FeedForwardNetwork()
inLayer = LinearLayer(size_x*size_x)
hiddenLayer = SigmoidLayer(hidden)
outLayer = LinearLayer(size_x*size_x)
net.addInputModule(inLayer)
net.addModule(hiddenLayer)
net.addOutputModule(outLayer)
in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)
net.addConnection(in_to_hidden)
net.addConnection(hidden_to_out)
net.sortModules()

print "Adding Samples"
ds = SupervisedDataSet(size_x*size_x, size_x*size_x)
for i in range(1000):
  data = numpy.random.randn(size_x*size_x)
  ds.addSample(data,data)

print "Training"
trainer = BackpropTrainer(net, ds)
for i in range(100):
  print trainer.train()

