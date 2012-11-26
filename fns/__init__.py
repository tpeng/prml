from numpy import exp
from numpy import frompyfunc
import numpy as np

# the simplest basis function
def identity(x):
  return x

def logistic(x):
  return 1 / (1 + exp(-x))

def apply_fn(fn, x):
  return frompyfunc(fn, 1, 1)(x)

def apply_fns(fns, x):
  return np.array([fn(x) for fn in fns]).T
