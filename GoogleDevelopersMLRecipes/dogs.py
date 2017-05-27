import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

grey_hieght = 28 + 4 * np.random.randn(greyhounds)
lab_hieght = 24 + 4 * np.random.randn(labs)

plt.hist([grey_hieght, lab_hieght], stacked=True, color=['r','b'])
plt.show()