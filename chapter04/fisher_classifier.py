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

class FisherClassifier():

  def __init__(self, class1, class2):
    self.class1 = class1
    self.class2 = class2

  def train(self):
    mean1 = np.mean(self.class1, axis=0)
    mean2 = np.mean(self.class2, axis=0)

    sw = np.dot((self.class1 - mean1).T, self.class1 -mean1) + np.dot((self.class2 - mean2).T, self.class2-mean2)
    self.w = np.dot(np.linalg.inv(sw), mean2 - mean1)

    print "vector of max weights",self.w

  def plot(self):
    plt.plot(np.dot(self.class1, self.w), [0]*self.class1.shape[0], 'bo')
    plt.plot(np.dot(self.class2, self.w), [0]*self.class2.shape[0], 'r-')
    plt.legend()
    plt.show()
