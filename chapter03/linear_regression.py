# linear model for regression in PRML chapter 03
# tpeng <pengtaoo@gmail.com>
# 2012/11/26
import numpy as np
from fns import apply_fns

class Regression():
  def __init__(self, _lambda, N, M, fns):
    self._lambda = _lambda
    self._N = N
    self._M = M
    self._fns = fns

  def fit(self, x, y):
    x0 = np.zeros((self._N, self._M), float)
    for i in range(self._N):
      for j in range(self._M):
        x0[i,j] = self._fns[j](j, x[i])
    S = np.dot(x0.T, x0)
    if S.size == 1:
      S += self._lambda
    else:
      S += self._lambda * np.identity(len(S))
    self.w = np.linalg.solve(S, np.dot(x0.T, y))
    self.b = np.mean(y) - np.dot(self.w.T, np.mean(x0, axis=0))

  def predict(self, xs):
    x0 = np.zeros((len(xs), self._M), float)
    for i in range(len(xs)):
      for j in range(self._M):
        x0[i,j] = self._fns[j](j, xs[i])
    v = np.dot(x0, self.w)
    return v + self.b
