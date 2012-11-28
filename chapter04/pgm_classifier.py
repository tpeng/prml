# Probabilistic Generative Model (PRML 4.2)
# tpeng <pengtaoo@gmail.com>
# 2012/11/28
import numpy as np
import math

def gaussian(x, mean, cov):
  xm = np.reshape((x - mean), (-1, 1))
  dim = len(mean)
  px = 1 / (math.pow(2 * math.pi, dim/2)) * 1 / math.sqrt(np.linalg.det(cov)) * math.exp(-(np.dot(np.dot(xm.T, np.linalg.inv(cov)), xm)) / 2)
  return px

class PGM():
  def __init__(self):
    self.means = []
    self.covs = []
    self.counts = []
    self.priors = []

  def fit(self, *params):
    for i, c in enumerate(params):
      self.means.append(params[i].mean(axis=0))
      self.covs.append(np.cov(params[i], rowvar=0))
      self.counts.append(len(params[i]))

    for i in range(len(params)):
      self.priors.append(self.counts[i] * 1.0 / sum(self.counts))

    self.cov = np.sum(map(lambda (w, v) : w*v, zip(self.priors, self.covs)), axis=0)

  def predict(self, x):
    probs = []
    for i in range(len(self.means)):
      cond_prob = gaussian(x, self.means[i], self.cov)
      probs.append(cond_prob * self.priors[i])
    return np.argmax(probs)
