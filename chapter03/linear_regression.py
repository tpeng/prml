# linear model for regression in PRML chapter 03
# tpeng <pengtaoo@gmail.com>
# 2012/11/26
import numpy as np
from fns import apply_fns

class Regression():
  def __init__(self, _lambda, fns):
    self._lambda = _lambda
    self._fns = fns

  def fit(self, x, y):
    x0 = apply_fns(self._fns, x)
    S = np.dot(x0.T, x0)
    if S.size == 1:
      S += self._lambda
    else:
      S += self._lambda * np.identity(len(S))
    self.w = np.linalg.solve(S, np.dot(x0.T, y))
    self.w0 = np.mean(y) - np.dot(self.w, np.mean(x0, axis=0))

    print self.w

  def predict(self, x, fns):
    x0 = apply_fns(fns, x)
    v = np.dot(x0, self.w)
    return v + self.w0
