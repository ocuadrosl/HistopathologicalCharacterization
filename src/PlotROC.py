import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import csv
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

file = open('/home/oscar/MEGA/post-doc/papers/cbms/results/high_low.csv', 'rt')

dataset = csv.reader(file, delimiter=",", quoting=csv.QUOTE_NONE)
x = list(dataset)
header = x[0]
values = x[1:len(x)]
values = np.array(values).astype('float')
target = values[:,len(values[0])-1]
values = values[:, 0:len(values[0])-1]

X = values
y = target
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

# shuffle and split training and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

#clf.fit(X_train, y_train)
clf.fit(X, y)
MLPClassifier(alpha=1e-05, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False,
              epsilon=1e-08, hidden_layer_sizes=(5, 2),
              learning_rate='constant', learning_rate_init=0.3,
              max_iter=200, momentum=0.2, n_iter_no_change=10,
              nesterovs_momentum=True, power_t=0.5, random_state=1,
              shuffle=True, solver='lbfgs', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)

print(y)
#y_pred = clf.predict(X_test)
y_pred = clf.predict(X)
print(y_pred)

n_classes = int(header[1])
#print(y_test.shape)
print(y_pred.shape)




tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

#fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
fpr, tpr, thresholds = metrics.roc_curve(y, y_pred)
tprs.append(interp(mean_fpr, fpr, tpr))
tprs[-1][0] = 0.0
roc_auc = auc(fpr, tpr)
aucs.append(roc_auc)
#plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (1, roc_auc))
#plt.show()
print(fpr, tpr, thresholds)



plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()









