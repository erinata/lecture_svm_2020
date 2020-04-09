import kfold_template
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles
from sklearn import svm

data, target  = make_circles(n_samples = 500, noise = 0.12)

plt.scatter(data[:,0], data[:,1], c=target)
plt.savefig("plot.png") 

r2_scores, accuracy_scores, confusion_matrices = kfold_template.run_kfold(5, data, target, svm.SVC(kernel="rbf", gamma=1), 1, 1)


print(r2_scores)
print(accuracy_scores)
for i in confusion_matrices:
	print(i)
