# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 11:20:13 2019

@author: Orin
"""

import numpy as np
from sklearn.svm import SVC


def classify_SVM(features_train, labels_train):

    clf = SVC(C=1000.0, kernel='rbf')
    clf.fit(features_train, labels_train)
    return clf