



import numpy as np
import matplotlib.pyplot as plt

mu_first = [-1, -1]
mu_second = [1, 1]
cov_first = [[1, -0.5], [-0.5, 1]]
cov_second = [[1, -0.5], [-0.5, 1]]

x1,y1 = np.random.multivariate_normal(mu_first, cov_first, 250).T
x2,y2 = np.random.multivariate_normal(mu_second, cov_second, 250).T

plt.plot(x1,y1, marker = '*',ls = '')
plt.plot(x2,y2,marker = 'o',ls = '')
plt.axis([-4 ,+4, -4 ,+4])
plt.show() 