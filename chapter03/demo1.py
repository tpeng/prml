import numpy as np
import pylab as pl
from chapter03.linear_regression import Regression
from fns import  identity

if __name__ == '__main__':
  np.random.seed(7)

  # 100 data point
  N = 100
  s = 10

  # the data to learn
  x = s * np.random.rand(N)
  w = np.random.rand()
  y = w * x + 0.1 * np.random.randn(N)

  # plot the raw data (sin + noise)
  pl.plot(x,y,'.r')

  # predict with regression
  x0 = np.linspace(min(x),max(x),500)

  m = Regression(0.5, [identity])
  m.fit(x, y)

  y0 = m.predict(x0, [identity])

  pl.plot(x0, y0)
  pl.show()
