import random
import re
import numpy as np
from chapter04.pgm_classifier import PGM

def read_data(N=110):
  f = open('..\dataset\\classification.txt', 'r')
  lines = map(lambda x: x.strip(), f.readlines())
  random.shuffle(lines)
  f.close()

  class1 = []
  class2 = []

  for line in lines[0:N]:
    elems = re.split("\s+",line)
    y = elems.pop()
    y = float(y.rstrip())

    xs = [float(elem) for elem in elems]

    if int(y) == 0:
      class1.append(xs)
    else:
      class2.append(xs)

  class1 = np.array(class1)
  class2 = np.array(class2)

  test_lines = [re.split("\s+", line) for line in lines[N:] if line]
  test_data = np.array([line[:1] for line in test_lines], dtype=np.float)
  test_labels = []

  for line in test_lines:
    y = line[-1]
    y = float(y.rstrip())
    test_labels.append(y)

  return class1, class2, test_data, test_labels

if __name__ == '__main__':
  c1, c2, tests, test_labels = read_data()
  m = PGM()

  # train
  m.fit(c1, c2)

  errors = 0
  # predict
  for i, t in enumerate(tests):
    y = m.predict(t)
    if y != test_labels[i]:
      errors += 1

  print '%s errors out of %s instance.' %(errors, len(tests))
  print 'accuracy %s' %((len(tests) - errors) * 1.0 / len(tests))