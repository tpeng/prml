# data from http://research.microsoft.com/en-us/um/people/cmbishop/prml/webdatasets/classification.txt
# described in PRML Appendix A.7
#
import re
from matplotlib.pyplot import figure
import pylab as pl
import numpy as np
from chapter04.fisher_classifier import FisherClassifier

def read_data():
  f = open('..\dataset\\classification.txt', 'r')
  lines = map(lambda x: x.strip(), f.readlines())
  f.close()

  class1 = []
  class2 = []

  for line in lines:
    elems = re.split("\s+",line)
    y = elems.pop()
    y = float(y.rstrip())

    xs = [float(elem) for elem in elems]

    if int(y) == 0:
      class1.append(xs)
    else:
      class2.append(xs)

  class1 = np.array(class1)
  class2 = np.array(class2)

  return class1, class2

class1, class2 = read_data()
# plot the raw data
figure(0)
pl.plot(class1[:,0], class1[:,1], 'xr')
pl.plot(class2[:,0], class2[:,1], 'ob')

figure(1)

m = FisherClassifier(class1, class2)
m.fit()

pl.plot(np.dot(class1, m.w), [0]*class1.shape[0], 'bo')
pl.plot(np.dot(class2, m.w), [0]*class2.shape[0], 'r-')
pl.show()
