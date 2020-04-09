import kfold_template
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles 
from sklearn import svm

import numpy as np

from mpl_toolkits.mplot3d import Axes3D

data, target  = make_circles(n_samples = 500, noise = 0.04)

# print(data)
# print(target)
plt.scatter(data[:,0], data[:,1], c=target)
plt.savefig("plot.png")


data1 = data[:,0].reshape((-1, 1)) 
data2 = data[:,1].reshape((-1, 1)) 
data3 = (data1**2 + data2**2)

data = np.hstack((data,data3))

# print(data)

fig = plt.figure()
axes = fig.add_subplot(111, projection = "3d")
axes.scatter(data1, data2, data3, c=target, depthshade=True)
# plt.savefig("plot3d.png")


# r2_scores, accuracy_scores, confusion_matrices = kfold_template.run_kfold(5, data, target, svm.SVC(kernel = "linear"), 1, 1)

# print(r2_scores)
# print(accuracy_scores)
# for i in confusion_matrices:
# 	print(i)


machine = svm.SVC(kernel = "linear")
machine.fit(data, target)
coeff = machine.coef_
intercept = machine.intercept_

# print(coeff)
print(coeff)
print(intercept)

data1, data2 = np.meshgrid(data1, data2) 

print(data1)
print(data2)

plane = -(coeff[0][0]*data1 + coeff[0][1]*data2 + intercept) / coeff[0][2]

axes1 = fig.gca(projection = '3d') 
axes1.plot_surface(data1, data2, plane, alpha = 0.01) 
plt.savefig("plot3d.png")


















# r2_scores, accuracy_scores, confusion_matrices = kfold_template.run_kfold(5, data, target, svm.SVC(kernel = "linear"), 1, 1)


# print(r2_scores)

# print(accuracy_scores)

# for i in confusion_matrices:
# 	print(i)