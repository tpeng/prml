# kmeans implementation in PRML chapter 09
# tpeng <pengtaoo@gmail.com>
# 2013/1/10
import numpy as np
import pylab as plt

def distance(x1, x2):
    return np.sum((x1-x2)**2)

class Kmeans():

    def __init__(self):
        self.centroids = []

    def fix(self, xs, k):
        prev_J= J = 0
        tol = 1e-4
        n,d = xs.shape
        # uk is the centroid of k
        uk = np.zeros((k, d), float)

        for i in range(k):
            uk[i] = xs[np.random.randint(0, n)]

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

            # fix rnk and minimize with uk
            # we select to the means of all the point belonging to centroid k
            for j in range(k):
                x = []
                for i in range(n):
                    if rnk[i, j] == 1:
                        x.append(xs[i])
                uk[j] = np.array(x).mean(axis=0)

            J = 0
            for i in range(n):
                for j in range(k):
                    J += rnk[i,j] * distance(xs[i], uk[j])

            if abs(J-prev_J) < tol:
                converged=True
            else:
                print J,
                prev_J = J

        print 'converged. '
        self.centroids = uk
        self.rnk = rnk

def read_data(name):
    xs = []
    for line in open(name):
        xs.append(map(float, line.rstrip().split()))
    return np.array(xs)

def plot(xs, centroids):
    plt.scatter(xs[:,0], xs[:,1])
    plt.scatter(centroids[:,0], centroids[:,1], color='r')
    plt.show()

if __name__ == '__main__':
    xs = read_data('../dataset/faithful.dat')
    kmeans = Kmeans()
    kmeans.fix(xs, 2)
    print kmeans.centroids
    plot(xs, kmeans.centroids)