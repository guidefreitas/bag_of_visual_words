import opf
import numpy

X = numpy.array(((1,1),(1,0),(0,1),(0,0),(2,2),(4,4),(8,8)))
y = numpy.array((2,1,1,0,4,16,64))

P = numpy.array(((4,4),(1,0),(0,0),(1,1),(2,2)))
print P
opf_cls = opf.Opf()
opf_cls.fit(X,y)
result = opf_cls.predict(P)
print result

accuracy, acc_time = opf_cls.accuracy(P)
print "Accuracy: ", accuracy, " Time: ", acc_time

