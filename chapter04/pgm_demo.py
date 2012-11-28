import random
import numpy as np
from chapter04.pgm_classifier import PGM

def read_data(N=120):
  f = open('..\\dataset\\bezdekIris.data', 'r')
  rawlines = [line.strip() for line in f.readlines()]
  random.shuffle(rawlines)
  f.close()

  lines = [line.split(",") for line in rawlines[0:N] if line]
  class1 = np.array([line[:4] for line in lines if line[-1] == 'Iris-setosa'], dtype=np.float)
  class2 = np.array([line[:4] for line in lines if line[-1] == 'Iris-versicolor'], dtype=np.float)
  class3 = np.array([line[:4] for line in lines if line[-1] == 'Iris-virginica'], dtype=np.float)

  test_lines = [line.split(",") for line in rawlines[N:] if line]
  test_data = np.array([line[:4] for line in test_lines], dtype=np.float)
  test_labels = []
  for line in test_lines:
    if line[-1] == 'Iris-setosa':
      test_labels.append(0)
    elif line[-1] == 'Iris-versicolor':
      test_labels.append(1)
    elif line[-1] == 'Iris-virginica':
      test_labels.append(2)

  return class1, class2, class3, test_data, test_labels

if __name__ == '__main__':
  c1, c2, c3, tests, test_labels = read_data()
  m = PGM()

  # train
  m.fit(c1, c2, c3)

  errors = 0
  # predict
  for i, t in enumerate(tests):
    y = m.predict(t)
    if y != test_labels[i]:
      errors += 1

  print '%s errors out of %s instance.' %(errors, len(tests))