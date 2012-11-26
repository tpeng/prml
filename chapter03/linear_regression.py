import numpy as np
import pylab as pl

# the simplest basis function
def identity(x):
  return x

def numpy_identity(x):
  return np.frompyfunc(identity, 1, 1)

class Regression():
  def __init__(self):
    pass

  def fit(self, x, y):
    numpy_identity(x)
    theta = x.T
    self.w = np.dot(1 / np.dot(theta.T, theta) * theta.T, y)
    self.w0 = np.mean(y) - np.mean(np.dot(self.w, x))

  def predict(self, x):
    pred = map(lambda x : self.w0 + identity(x), x)
    return np.array(pred)

if __name__ == '__main__':
  np.random.seed(5)

  # 100 data point
  N = 100
  s = 10

  # the data to learn
  x = s * np.random.rand(N)
  w = np.random.rand()
  y = w * x + 0.5 * np.random.randn(N)

  # plot the raw data (sin + noise)
  pl.plot(x,y,'.r')

  # predict with regression
  x0 = np.linspace(min(x),max(x),500)

  m = Regression()
  m.fit(x, y)

  y0 = m.predict(x0)

  pl.plot(x0, y0)
  pl.show()


