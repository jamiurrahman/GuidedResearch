# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 11:20:13 2019

@author: Orin
"""
import numpy as np
from sklearn.naive_bayes import GaussianNB


def classify_NB(features_train, labels_train):
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    return clf
