# linear regression use abalone data <http://archive.ics.uci.edu/ml/datasets/Abalone>
# 2012/11/27
import re
import numpy as np
import pylab as pl

#
# Name / Data Type / Measurement Unit / Description
# -----------------------------
# Sex / nominal / -- / M, F, and I (infant)
# Length / continuous / mm / Longest shell measurement
# Diameter	/ continuous / mm / perpendicular to length
# Height / continuous / mm / with meat in shell
# Whole weight / continuous / grams / whole abalone
# Shucked weight / continuous	 / grams / weight of meat
# Viscera weight / continuous / grams / gut weight (after bleeding)
# Shell weight / continuous / grams / after being dried
#
# Rings / integer / -- / +1.5 gives the age in years
from chapter03.linear_regression import Regression

def read_data():
  f = open('..\dataset\\abalone.data', 'r')
  lines = map(lambda x: x.strip(), f.readlines())
  f.close()

#  random.shuffle(lines)

  #
  xs = np.zeros([4177,8],dtype=float)
  ys = np.zeros([4177,1],dtype=int)
  ln = 0

  for line in lines:
    elems = re.split(",",line)
    y = elems.pop()
    y = int(y.rstrip())
    x = []
    if elems[0] is 'M':
      sex_v = 1
    elif elems[0] is 'F':
      sex_v = -1
    else:
      sex_v = 0

    x.append(sex_v)
    x.extend([float(elem) for elem in elems[1:]])

    # Now set the values in the array:
    ys[ln] = y
    xs[ln] = x

    ln += 1

  return ys, xs

if __name__ == '__main__':
  ys, xs = read_data()
  N = 4000
  M = 8 # from sex to shell height

  # simple basis functions
  fns = [lambda j, x : x[j]] * M

  # train
  model = Regression(5, N, M, fns)
  model.fit(xs[0:N], ys[0:N])

  # predict
  predict_ys = model.predict(xs[N:])

  # plot the result
  x0 = range(0, len(predict_ys))
  pl.plot(x0, ys[N:], '-r')
  pl.plot(x0, predict_ys, '-')
  pl.show()






