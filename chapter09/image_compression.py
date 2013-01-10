from PIL import Image
from chapter09.kmeans import Kmeans
import numpy as np

if __name__ == '__main__':
    image = Image.open('Earth.bmp')
    image.convert('RGB')
    pixels = image.load()
    width, height = image.size
    print 'mode: ', image.mode

    # try data = np.asarray(image)
    xs = []
    for x in range(width):
        for y in range(height):
            xs.append(pixels[x,y][0:3])
    print xs

    kmeans = Kmeans()
    kmeans.fix(np.array(xs), 4)
    print kmeans.centroids
    print kmeans.rnk.shape

    for x in range(width):
        for y in range(height):
#            print kmeans.rnk[x*y].argmax(), kmeans.centroids[kmeans.rnk[x*y].argmax()]
            pixels[x,y] = tuple(map(int, kmeans.centroids[kmeans.rnk[x+width*y].argmax()]))
            print pixels[x,y]

    image.save('new.png')