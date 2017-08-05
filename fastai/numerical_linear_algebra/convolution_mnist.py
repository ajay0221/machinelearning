import numpy as np
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')
print mnist.keys()

print mnist['data'].shape, mnist['target'].shape

images = np.reshape(mnist['data'], (70000, 28, 28))
labels = mnist['target'].astype(int)

n = len(images)

print images.shape, labels.shape

images = images/255
plot(images[0])

