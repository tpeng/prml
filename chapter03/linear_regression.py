import numpy as np
import pylab as pl

class Regression():
  def __init__(self, x, y):
    self._x = x
    self._y = y

  def predict(self, x):
    return [1] * len(x)

if __name__ == '__main__':
  np.random.seed(5)

  # 100 data point
  N = 1000
  s = 10

  # the data to learn
  x = s * np.random.rand(N)
  y = np.sin(x) + 0.1*np.random.randn(N)

  # plot the raw data (sin + noise)
  pl.plot(x,y,'.r')

  # predict with regression
  x0 = np.linspace(min(x),max(x),500)

  m = Regression(x, y)
  y0 = m.predict(x0)

  pl.plot(x0, y0)
  pl.show()


