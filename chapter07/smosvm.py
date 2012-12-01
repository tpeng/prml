#
# python port of https://github.com/karpathy/svmjs
# tpeng <pengtaoo@gmail.com>
# 2012/11/30
#
import random
import re
import numpy as np
import pylab as pl

def linearkernel(x1, x2):
  return np.dot(x1, x2)

def rbfkernel(x1, x2, _lambda=0.5):
  s = 0
  for i in range(len(x1)):
    s += (x1[i] - x2[i])**2
  return np.exp(-s/(2*_lambda*_lambda))

class SMO():
  def __init__(self, data, labels):
    self.C = 1.0
    self.tol = 1e-4
    self.maxiter = 1000
    self.kernel = linearkernel
    self.numpasses = 10
    self.alpha = np.array([0] * len(data))
    self.b = 0.0
    self.data = data
    self.labels = labels

  def train(self):
    iter = 0
    passes = 0
    N = len(self.alpha)

    while passes < self.numpasses and iter < self.maxiter:
      alpha_changed = 0

      for i in range(N):
        Ei = self.margin_one(self.data[i]) - self.labels[i]

        if (self.labels[i] * Ei < -self.tol and self.alpha[i] < self.C) or \
           (self.labels[i] * Ei > self.tol and self.alpha[i] > 0):
          j = i
          while j == i:
            j = np.random.randint(0, N)
          Ej = self.margin_one(self.data[j]) - self.labels[j]

          # calculate the L and H bounds for j to ensure we're in [0, C] * [0, C] box
          ai = self.alpha[i]
          aj = self.alpha[j]

          if self.labels[i] == self.labels[j]:
            L = max(0, ai+aj-self.C)
            H = min(self.C, ai+aj)
          else:
            L = max(0, aj-ai)
            H = min(self.C, self.C+aj-ai)

          if abs(L-H) < 1e-4:
            continue

          eta = 2*self.kernel(self.data[i], self.data[j]) - \
                self.kernel(self.data[i], self.data[i]) - \
                self.kernel(self.data[j], self.data[j])
          if eta >= 0:
            continue

          # compute newaj and newai
          newaj = aj - self.labels[j] * (Ei - Ej) / eta
          if newaj > H:
            newaj = H
          if newaj <L:
            newaj = L
          if abs(aj -newaj) < 1e-4:
            continue
          self.alpha[j] = newaj

          newai = ai + self.labels[i] * self.labels[j] * (aj -newaj)
          self.alpha[i] = newai

          # update bias
          b1 = self.b - Ei - self.labels[i]*(newai-ai)*self.kernel(self.data[i], self.data[i]) - \
               self.labels[j] * (newaj-aj)*self.kernel(self.data[i], self.data[j])

          b2 = self.b - Ej - self.labels[i]*(newai-ai)*self.kernel(self.data[i],self.data[j]) - \
               self.labels[j]*(newaj-aj)*self.kernel(self.data[j],self.data[j])

          self.b = (b1+b2)/ 2

          if 0 < newai < self.C:
            self.b = b1
          if 0 < newaj < self.C:
            self.b = b2

          alpha_changed += 1
      # end if
    # end for
      iter += 1
      print 'iter number %d, alpha_changed %d' %(iter, alpha_changed)

      if alpha_changed == 0:
        passes += 1
      else:
        passes = 0
  # end  while

  def margin_one(self, xs):
    f = self.b
    for i in range(len(self.data)):
      f += self.alpha[i]*self.labels[i] * self.kernel(xs, self.data[i])
    return f

  def predict(self, xs):
    if self.margin_one(xs) >= 0:
      return 1
    else:
      return -1

def read_data(N=100):
#  f = open('..\dataset\\mla\\testSet.txt', 'r')
  f = open('..\dataset\\classification.txt', 'r')
#
  lines = map(lambda x: x.strip(), f.readlines())
  random.shuffle(lines)
  f.close()

  data = []
  labels = []

  for line in lines[0:N]:
    elems = re.split("\s+", line)
    y = elems.pop()
    y = int(float(y.rstrip()))
    if y == 0:
      y = -1

    xs = [float(elem) for elem in elems]

    data.append(xs)
    labels.append(y)

  test_data = []
  test_labels = []

  for line in lines[N:]:
    elems = re.split("\s+", line)
    y = elems.pop()
    y = int(float(y.rstrip()))
    if y == 0:
      y = -1

    xs = [float(elem) for elem in elems]
    test_data.append(xs)
    test_labels.append(y)

  return data, labels, test_data, test_labels

if __name__ == '__main__':
  data, labels, tests, test_labels = read_data()

  c1 = np.array([data[i] for i, a in enumerate(labels) if a == 1])
  c2 = np.array([data[i] for i, a in enumerate(labels) if a == -1])

  pl.plot(c1[:,0], c1[:,1], 'xr')
  pl.plot(c2[:,0], c2[:,1], 'ob')

  smo = SMO(data, labels)
  smo.train()

  # plot the support vectors
  support_vectors = np.array([data[i] for i, a in enumerate(smo.alpha) if a > 0])
  pl.plot(support_vectors[:,0], support_vectors[:,1], 'oc', markersize=15)
  pl.show()

  errors = 0
  # predict
  for i, t in enumerate(tests):
    y = smo.predict(t)
    if y != test_labels[i]:
      print 'predict %s but it is %s' %(y, test_labels[i])
      errors += 1

  print '%s errors out of %s instance.' %(errors, len(tests))
  print 'accuracy %s' %((len(tests) - errors) * 1.0 / len(tests))
