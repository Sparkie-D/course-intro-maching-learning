from collections import Counter
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

def load_data ():
    feature, label = datasets.load_iris(return_X_y=True)
    print(feature.shape)
    print(label.shape)
    return feature, label

feature, label = load_data()
feature_train, feature_test, label_train, label_test = train_test_split(feature, label, test_size = 0.2, random_state = 0)
# print(Counter(label_train))
# print(Counter(label_test))

# question 2
classifier = GaussianNB()
classifier = classifier.fit(feature_train, label_train)
print(classifier.predict(feature_test))
print(classifier.score(feature_test, label_test))

# question 3
classifier.class_prior_ = [1./3, 1./3, 1./3]
classifier = classifier.fit(feature_train, label_train)
print(classifier.predict(feature_test))
print(classifier.score(feature_test, label_test))