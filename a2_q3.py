import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn import tree
from sklearn.datasets import load_digits

digits = load_digits()

print(digits.data.shape)
print("Label Data Shape ", digits.target.shape)

# view the images
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
 plt.subplot(1, 5, index + 1)
 plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
 plt.title('Training: %i\n' % label, fontsize = 20)
plt.show()

# divide up the data into training/test
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, train_size=0.8)


print("Here is the raw data of one of the training images: ", X_train[0])
print("Here is image as an 8x8 matrix: \n", np.reshape(X_train[0], (8,8)))
print("Here are some classes: ", y_test[0:10])
# add code here to do the classification


#
# softmax regression
#

print("\nbegin softmax regression")
softmax_reg = linear_model.LogisticRegression(solver='lbfgs', C=10, random_state=42, multi_class='multinomial')
softmax_reg.fit(X_train,y_train)

train_score = softmax_reg.score(X_train, y_train)
test_score = softmax_reg.score(X_test, y_test)

print("logistic regression training accuracy: ", train_score)
print("logistic regression test accuracy: ", test_score)

# confusion matrix using graphviz
# temp print("\nbegin confusion matrix graphviz")
# temp predictions = softmax_reg.predict(X_test)
# temp cm = metrics.confusion_matrix(y_test, predictions)
# temp plt.figure(figsize=(9,9))
# temp sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
# temp plt.ylabel('Actual label');
# temp plt.xlabel('Predicted label');
# temp all_sample_title = 'Accuracy Score: {0}'.format(test_score)
# temp plt.title(all_sample_title, size = 15);
# temp plt.show()

# confusion matrix plain
print("\nbegin plain confusion matrix")
disp = metrics.plot_confusion_matrix(softmax_reg, X_test, y_test, display_labels=None, cmap=plt.cm.Blues)
disp.ax_.set_title('Accuracy Score: {0}'.format(test_score))
print('Accuracy Score: {0}'.format(test_score))
print(disp.confusion_matrix)
plt.show()


print("done")
