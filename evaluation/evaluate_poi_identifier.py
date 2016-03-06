#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here
import numpy
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, precision_score, recall_score
from time import time

features = numpy.array(features)
labels = numpy.array(labels)

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
    features, labels, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()

# print 'train features:\n', features_train
# print 'train labels:\n', labels_train

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time() - t0, 3), "s"

t0 = time()
prediction = clf.predict(features_test)
print "prediction time:", round(time() - t0, 3), "s"

print 'predicted poi', sum(prediction)
print 'prediction total', len(prediction)

zero_prediction = numpy.zeros_like(prediction)

accuracy = accuracy_score(labels_test, prediction)
print 'accuracy:', accuracy
print 'accuracy with all 0:', accuracy_score(labels_test, zero_prediction)
print 'precision score:', precision_score(labels_test, prediction)
print 'recall score:', recall_score(labels_test, prediction)

preds = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
trues = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

print 'true pos:', sum([1 for i in range(len(preds)) if preds[i] == 1 and trues[i] == 1])
print 'true neg:', sum([1 for i in range(len(preds)) if preds[i] == 0 and trues[i] == 0])
print 'false pos:', sum([1 for i in range(len(preds)) if preds[i] == 1 and trues[i] == 0])
print 'false neg:', sum([1 for i in range(len(preds)) if preds[i] == 0 and trues[i] == 1])
print 'precision:', precision_score(trues, preds)
print 'recall:', recall_score(trues, preds)
