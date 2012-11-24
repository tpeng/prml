from __future__ import division
import numpy as np
import math

def read_data():
  f = open('dataset\\bezdekIris.data', 'r')
  lines = [line.strip() for line in f.readlines()]
  f.close()

  lines = [line.split(",") for line in lines if line]
  data = np.array([line[:4] for line in lines if line], dtype=np.float)
  class1 = np.array([line[:4] for line in lines if line[-1] == 'Iris-setosa'], dtype=np.float)
  class2 = np.array([line[:4] for line in lines if line[-1] == 'Iris-versicolor'], dtype=np.float)
  class3 = np.array([line[:4] for line in lines if line[-1] == 'Iris-virginica'], dtype=np.float)

  labels = []
  for line in lines:
    strt = line.pop()
    labels.append(strt)

  labels = [line.split(',') for line in labels if line]

  t = np.zeros(shape=(150, 3))

  for i in xrange(len(data)):
    if labels[i] == ['Iris-setosa']: t[i][0] = 1
    elif labels[i] == ['Iris-virginica']: t[i][1] = 1
    elif labels[i] == ['Iris-versicolor']: t[i][2] = 1

  return class1, class2, class3, data, t

def gaussian(x, mean, cov):
  xm = np.reshape((x - mean), (-1, 1))
  # since dim(x) is 4, so p/2 equal to 2
  px = 1 / (math.pow(2 * math.pi, 2)) * 1 / math.sqrt(np.linalg.det(cov)) * math.exp(-(np.dot(np.dot(xm.T, np.linalg.inv(cov)), xm)) / 2)
  return px

def main():
  class1, class2, class3, data, t = read_data()
  count = np.zeros(shape=(150,1))
  t_assigned = np.zeros(shape=(150, 3))

  mean1 = class1.mean(axis=0)
  mean2 = class2.mean(axis=0)
  mean3 = class3.mean(axis=0)

  cov1 = np.cov(class1, rowvar=0)
  cov2 = np.cov(class2, rowvar=0)
  cov3 = np.cov(class3, rowvar=0)

  # compute gaussian likelihood functions p(x|Ck) and prior(Ck) for each class
  for i in xrange(len(data)):
    # prior of class1, class2, class3 are all 1/3
    px1 = gaussian(data[i], mean1, cov1) * 1 / 3.0
    px2 = gaussian(data[i], mean2, cov2) * 1 / 3.0
    px3 = gaussian(data[i], mean3, cov3) * 1 / 3.0

    # compute posterior probability p(Ck|x) assuming that p(x|Ck) is gaussian and the entire expression is wrapped by sigmoid function
    pc1 = px1 / (px2 + px3)
    pc2 = px2 / (px1 + px3)
    pc3 = px3 / (px1 + px2)

    if pc1 > pc2 and pc1 > pc3: t_assigned[i][0] = 1
    elif pc3 > pc1 and pc3 > pc2: t_assigned[i][1] = 1
    elif pc2 > pc1 and pc2 > pc3: t_assigned[i][2] = 1

    # count the number of misclassifications
    for j in xrange(3):
      if t[i][j] - t_assigned[i][j] != 0: count[i] = 1

  print "number of misclassifications %d out of %d instances" %(sum(count), len(data))

main()