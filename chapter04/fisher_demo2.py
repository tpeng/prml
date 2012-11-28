# http://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data
import re
from matplotlib.pyplot import figure
import numpy as np
import pylab as pl
from chapter04.fisher_classifier import FisherClassifier

def read_data():
  f = open('..\dataset\\haberman.data', 'r')
  lines = map(lambda x: x.strip(), f.readlines())
  f.close()

  class1 = []
  class2 = []

  for line in lines:
    elems = re.split(",",line)
    y = elems.pop()
    y = float(y.rstrip())

    xs = [float(elem) for elem in elems]

    if int(y) == 1:
      class1.append(xs)
    else:
      class2.append(xs)

  class1 = np.array(class1)
  class2 = np.array(class2)

  return class1, class2

class1, class2 = read_data()

class1, class2 = read_data()
m = FisherClassifier(class1, class2)
m.fit()

# plot the boundary
figure(1)
pl.plot(np.dot(class1, m.w), [0]*class1.shape[0], 'bo')
pl.plot(np.dot(class2, m.w), [0]*class2.shape[0], 'rx')

pl.show()

