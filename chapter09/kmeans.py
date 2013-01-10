import numpy as np
import pylab as plt

def distance(x1, x2):
    return np.sum((x1-x2)**2)

class Kmeans():

    def fix(self, xs, k):
        n,d = xs.shape
        # uk is the centroid of k
        uk = np.zeros((k, d), float)

        converged = False
        while not converged:
            # fix uk and minimize with rnk
            # we select nearest k for point n to minimize it
            distances = np.zeros((n,k), float)

            for i in range(n):
                for j in range(k):
                    distances[i,j] = distance(xs[i], uk[j])

            rnk = np.zeros((n,k), float)
            for (i, j) in enumerate(distances.argmin(axis=1)):
                rnk[i, j] = 1

            print rnk[0:10]
            # fix rnk and minimize with uk
            # we select to the means of all the point belonging to centroid k
            for j in range(k):
                uk[j] = (xs[rnk == 1]).mean()

            print uk

def read_data(name):
    xs = []
    for line in open(name):
        xs.append(map(float, line.rstrip().split()))
    return np.array(xs)

def plot(xs):
    plt.scatter(xs[:,0], xs[:,1])
    plt.show()

if __name__ == '__main__':
    xs = read_data('../dataset/faithful.dat')
    x1 = [1, 2]
    x2 = [3, 3]
    #    plot(xs)
    kmeans = Kmeans()
    kmeans.fix(xs, 4)