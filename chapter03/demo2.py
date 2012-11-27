# use linear model to fit sin(x) + noise
import numpy as np
import pylab as pl
from chapter03.linear_regression import Regression

# simple basis functions
fns = [lambda j, x : x, lambda j, x : x**2, lambda j, x: x**3, lambda j, x: x**4]

if __name__ == '__main__':
  np.random.seed(7)

  # 100 data point
  N = 100
  M = len(fns)
  s = 10

  # the data to learn
  x = s * np.random.rand(N)
  w = np.random.rand()
  y = np.sin(x) + 0.1 * np.random.randn(N)

  # plot the raw data (sin + noise)
  pl.plot(x,y,'.r')

  # predict with regression
  x0 = np.linspace(min(x),max(x),500)

  m = Regression(0.5, N, M, fns)
  m.fit(x, y)

  y0 = m.predict(x0)

  pl.plot(x0, y0)
  pl.show()

