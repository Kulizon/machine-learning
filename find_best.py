from sklearn.datasets import load_iris
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

iris = load_iris()
data_set = iris.data[0:len(iris.target)-20,:]
labels = iris.target[0:len(iris.target)-20]
test_data_set = iris.data[-20:,:]
test_labels = iris.target[-20:]

cls_dict = {
    "lr": LinearRegression(),
    "knn": KNeighborsClassifier(),
    "svm": SVC(),
    "dt": DecisionTreeClassifier(),
    "nb": GaussianNB(),
    "qda": QuadraticDiscriminantAnalysis()
}

import itertools
best_acc = 0
best_comb = []
for comb in itertools.combinations(cls_dict.keys(), 3):
    # fresh instances
    classifiers = [
        LinearRegression() if k == "lr" else
        KNeighborsClassifier() if k == "knn" else
        SVC() if k == "svm" else
        DecisionTreeClassifier() if k == "dt" else
        GaussianNB() if k == "nb" else
        QuadraticDiscriminantAnalysis()
        for k in comb
    ]
    for c in classifiers:
        c.fit(data_set, labels)
    
    output = []
    for classifier in classifiers:
        pred = classifier.predict(data_set)
        # Linear regression can output continuous instead of classification labels, so we round it
        if type(classifier) == LinearRegression:
            pred = np.round(pred).astype(int)
        output.append(pred)
    output = np.array(output).T
    
    stacked_classifier = DecisionTreeClassifier()
    stacked_classifier.fit(output, labels)
    
    test_set = []
    for classifier in classifiers:
        pred = classifier.predict(test_data_set)
        if type(classifier) == LinearRegression:
            pred = np.round(pred).astype(int)
        test_set.append(pred)
    test_set = np.array(test_set).T
    predicted = stacked_classifier.predict(test_set)
    acc = accuracy_score(test_labels, predicted)
    if acc > best_acc:
        best_acc = acc
        best_comb = comb
print(best_acc, best_comb)
