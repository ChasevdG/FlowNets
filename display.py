import matplotlib.pyplot as plt
import numpy as np

a = np.load('cov.csv.npy')
n= len(a)

for i in range(n):
    print(a[i,n-i-1])
plt.imshow(a)
plt.show()
