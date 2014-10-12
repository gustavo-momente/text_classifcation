#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Usuário'

from sklearn.linear_model import Perceptron
from sklearn import svm
from sklearn import cross_validation
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB


class learn_by_svm:
    def __init__(self, X=None, Y=None, path=r"..\..\svm_dump.pkl", C=0.5, kernel='linear', fit=True):
        if X is None or Y is None:
            self.clf = joblib.load(path)
        else:
            self.clf = svm.SVC(kernel=kernel, C=C, cache_size=500, class_weight='auto')
            if fit:
                self.clf.fit(X, Y)
                self.dump(path)

    def predict(self, X):
        return self.clf.predict(X)

    def cross_val(self, X, Y, n, cpus=6):
        return cross_validation.cross_val_score(self.clf, X, Y, cv=n, n_jobs=cpus, scoring='f1')

    def dump(self, path=r"..\..\svm_dump.pkl"):
        joblib.dump(self.clf, path)


class learn_by_perceptron:
    def __init__(self, X=None, Y=None, path=r"..\..\per_dump.pkl", penalty='l1', alpha=0.00001, fit=True):
        if X is None or Y is None:
            self.clf = joblib.load(path)
        else:
            self.clf = Perceptron(penalty=penalty, alpha=alpha, n_jobs=6, class_weight='auto', shuffle=True)
            if fit:
                self.clf.fit(X, Y)
                self.dump(path)

    def predict(self, X):
        return self.clf.predict(X)

    def cross_val(self, X, Y, n, cpus=6):
        return cross_validation.cross_val_score(self.clf, X, Y, cv=n, n_jobs=cpus, scoring='f1')

    def dump(self, path=r"..\..\svm_dump.pkl"):
        joblib.dump(self.clf, path)


class learn_by_MNbayes:
    def __init__(self, X=None, Y=None, path=r"..\..\MNB_dump.pkl", alpha=1, fit=True):
        if X is None or Y is None:
            self.clf = joblib.load(path)
        else:
            self.clf = MultinomialNB(alpha=alpha)
            if fit:
                self.clf.fit(X, Y)
                self.dump(path)

    def predict(self, X):
        return self.clf.predict(X)

    def cross_val(self, X, Y, n, cpus=6):
        return cross_validation.cross_val_score(self.clf, X, Y, cv=n, n_jobs=cpus, scoring='f1')

    def dump(self, path=r"..\..\svm_dump.pkl"):
        joblib.dump(self.clf, path)


class learn_by_svc:
    def __init__(self, X=None, Y=None, path=r"..\..\svc_dump.pkl", C=1, fit=True):
        if X is None or Y is None:
            self.clf = joblib.load(path)
        else:
            n_samples, n_features = X.shape
            self.clf = svm.LinearSVC(C=C, dual=(n_features >= n_samples), class_weight='auto')
            if fit:
                self.clf.fit(X, Y)
                self.dump(path)

    def predict(self, X):
        return self.clf.predict(X)

    def cross_val(self, X, Y, n, cpus=6):
        return cross_validation.cross_val_score(self.clf, X, Y, cv=n, n_jobs=cpus, scoring='f1')

    def dump(self, path=r"..\..\svc_dump.pkl"):
        joblib.dump(self.clf, path)