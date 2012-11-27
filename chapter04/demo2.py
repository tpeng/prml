# http://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data
import re
from matplotlib.pyplot import figure
import numpy as np
import pylab as pl

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

mean1 = np.mean(class1, axis=0)
mean2 = np.mean(class2, axis=0)

sw = np.dot((class1 - mean1).T, class1 -mean1) + np.dot((class2 - mean2).T, class2-mean2)
w = np.dot(np.linalg.inv(sw), mean2 - mean1)

print "vector of max weights", w

# plot the boundary
figure(1)
pl.plot(np.dot(class1, w), [0]*class1.shape[0], 'bo')
pl.plot(np.dot(class2, w), [0]*class2.shape[0], 'rx')

pl.show()

