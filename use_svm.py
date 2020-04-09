import kfold_template
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn import svm


# Data generating
data, target = make_blobs(n_samples=400, centers=2, cluster_std=1, random_state=0)


print(data)
print(target)
plt.scatter(data[:,0], data[:,1], c=target)
plt.savefig("plot.png")


r2_scores, accuracy_scores, confusion_matrices = kfold_template.run_kfold(5, data, target, svm.SVC(kernel="linear"), 1, 1)

print(r2_scores)
print(accuracy_scores)
for i in confusion_matrices:
	print(i)





