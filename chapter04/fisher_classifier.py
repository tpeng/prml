# Fisher's linear discriminant
# tpeng <pengtaoo@gmail.com>
# 2012/11/27
import numpy as np

class FisherClassifier():

  def __init__(self, class1, class2):
    self.class1 = class1
    self.class2 = class2

  def fit(self):
    mean1 = np.mean(self.class1, axis=0)
    mean2 = np.mean(self.class2, axis=0)

    sw = np.dot((self.class1 - mean1).T, self.class1 -mean1) + np.dot((self.class2 - mean2).T, self.class2-mean2)
    self.w = np.dot(np.linalg.inv(sw), mean2 - mean1)

    m = (len(self.class1) * mean1 + len(self.class2) * mean2) / (len(self.class1) + len(self.class2)) # 4.36
    self.w0 = -np.dot(self.w.T, m)   # 4.34

  def predict(self, x):
    return np.dot(self.w.T, x) + self.w0