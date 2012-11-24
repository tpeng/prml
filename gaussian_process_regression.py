from __future__ import division
import numpy as np

x = [-1.5, -1.0, -0.75, -0.40, -0.25, 0.00]

class RBF():
  def __init__(self, Lambda):
    self._lambda = Lambda

  def compute(self, x, x2):
    X = np.matrix(x)
    X2 = np.matrix(x2)
    sq_norm = np.dot(X.T, X) + np.dot(X2.T, X) - 2*X*X2.T
    return np.exp(-sq_norm / self._lambda**2)

if __name__ == '__main__':
  rbf = RBF(1)
  X = np.matrix(x)
  print np.dot(X, X.T)
#  print rbf.compute(x,x)