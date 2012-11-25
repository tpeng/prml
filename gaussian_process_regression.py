#
# simple regression with Gaussian Process
# tpeng <pengtaoo@gmail.com>
# 2012/11/25
#
import pylab as pl
import numpy as np

# the Gaussian kernel
def rbf(x1, x2, _lambda=1.0):
  d = (x1-x2)**2 / _lambda
  return np.exp(-0.5*d)

class GP():
  def __init__(self, kn, beta=1.0):
    self.kn = kn
    self.beta = beta

  # compute the Gram matrix of k(x1, xn)
  def _K(self, x1, x2):
    # x1 and x2 have same dimension
    N = x1.shape[0]
    M = x2.shape[0]
    K = np.zeros((N,M), float)
    for i in range(0, N):
      for j in range(0, M):
        K[i, j] = self.kn(x1[i], x2[j])
    return K

  def fit(self, x, y):
    self._x = x
    K = self._K(x, x)
    # plus the noise
    K = K + self.beta * np.identity(len(K))
    self.alpha = np.linalg.solve(K, y)

  def predict(self, x):
    ks = self._K(x, self._x)
    return ks.dot(self.alpha)

if __name__ == '__main__':
  np.random.seed(5)

  # 100 data point
  N = 1000
  s = 10

  x = s * np.random.rand(N)
  y = np.sin(x) + 0.1*np.random.randn(N)

  # plot the raw data (sin + noise)
  pl.plot(x,y,'.r')

  # do the regression
  gp = GP(rbf)
  gp.fit(x, y)

  x0 = np.linspace(min(x),max(x),500)
  mean = gp.predict(x0)

  pl.plot(x0, mean)

  # show the prediction
  pl.show()

