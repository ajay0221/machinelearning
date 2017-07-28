import numpy as np

#Person x Demand quantity of foodstuff
A = np.array([[6, 5, 3, 1],
              [3, 6, 2, 2],
              [3, 4, 3, 1]])

#Prices of quantitis in Shops S1 and S2 x Shop
B = np.array([[1.5, 1.0],
              [2.0, 2.5],
              [5.0, 4.5],
              [16.0, 17.0]])

print np.matmul(A, B)
