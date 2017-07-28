import numpy as np

#Sources x Destination
A = np.array([[0.9, 0.07, 0.02, 0.01],
              [0, 0.93, 0.05, 0.02],
              [0, 0, 0.85, 0.15],
              [0, 0, 0, 1.0]])

#Source x 1
X = np.array([0.85, 0.1, 0.05, 0])

print np.matmul(np.transpose(A), np.transpose(X))
print np.matmul(X, A)
