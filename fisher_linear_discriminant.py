from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def read_data():
  f=open('dataset\\bezdekIris.data', 'r')
  lines = [line.strip() for line in f.readlines()]
  f.close()

  lines = [line.split(',') for line in lines if line]
  class1 = np.array([line[:4] for line in lines if line[-1] == 'Iris-setosa'], dtype=np.float)
  class2 = np.array([line[:4] for line in lines if line[-1] != 'Iris-setosa'], dtype=np.float)
  return class1, class2

def main():
  class1, class2 = read_data()
  mean1 = np.mean(class1, axis=0)
  mean2 = np.mean(class2, axis=0)

  sw = np.dot((class1 - mean1).T, class1 -mean1) + np.dot((class2 - mean2).T, class2-mean2)
  w = np.dot(np.linalg.inv(sw), mean2 - mean1)

  print "vector of max weights", w

  plt.plot(np.dot(class1, w), [0]*class1.shape[0], 'bx', label='iris-setosa')
  plt.plot(np.dot(class2, w), [0]*class2.shape[0], 'go', label='iris-versicolor and iris-virginica')
  plt.legend()
  plt.show()

if __name__ == '__main__':
  main()