# http://ftp.ics.uci.edu/pub/machine-learning-databases/iris/bezdekIris.data
import numpy as np
import pylab as pl
import random
from chapter04.fisher_classifier import FisherClassifier

def read_data(N=100):
  f=open('..\\dataset\\bezdekIris.data', 'r')
  lines = [line.strip() for line in f.readlines()]
  random.shuffle(lines)
  f.close()

  lines = [line.split(',') for line in lines if line]
  class1 = np.array([line[:4] for line in lines[0:N] if line[-1] == 'Iris-setosa'], dtype=np.float)
  class2 = np.array([line[:4] for line in lines[0:N] if line[-1] != 'Iris-setosa'], dtype=np.float)

  class1_tests = np.array([line[:4] for line in lines[N:] if line[-1] == 'Iris-setosa'], dtype=np.float)
  class2_tests = np.array([line[:4] for line in lines[N:] if line[-1] != 'Iris-setosa'], dtype=np.float)
  return class1, class2, class1_tests, class2_tests

if __name__ == '__main__':
  c1, c2, c1test, c2test = read_data()
  m = FisherClassifier(c1, c2)
  m.fit()

  pl.plot(np.dot(c1, m.w), [0]*c1.shape[0], 'bo')
  pl.plot(np.dot(c2, m.w), [0]*c2.shape[0], 'r-')
  pl.show()

  error = 0
  for i in range(len(c1test)):
    y = m.predict(c1test[i])
    if y > 0:
      error += 1

  for i in range(len(c2test)):
    y = m.predict(c2test[i])
    if y < 0:
      error += 1

  print '%s errors out of %s instances' %(error, (len(c1test) + len(c2test)))


