#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime


__author__ = 'Usuário'

import gensim.matutils
import numpy as np
from Parser import *
from Filter import *
from Cleaner import *
from Learners import *
from stopwords import *
import timeit
import sklearn
import sklearn.feature_extraction.text as txtTools #.TfidfTransformer
import os.path
import nltk

class SklearnTfIdf:

    def __init__(self, corpus, idf=True, only_presence=False, norm=None):

        self.data_sparse = gensim.matutils.corpus2csc(corpus)
        self.tf_idf_transformer = txtTools.TfidfTransformer(use_idf=idf, norm=norm)
        self.data = self.tf_idf_transformer.fit_transform(self.data_sparse.T)
        # self.data = self.tf_idf_transformer.transform(self.data_sparce.T)
        self.only_presence = only_presence
        if self.only_presence:
            sklearn.preprocessing.binarize(self.data, copy=False)
            # sklearn.preprocessing.Normalizer(copy=False).transform(self.data)



    def get_transformer(self):
        return self.tf_idf_transformer

    def get_features(self):
        return self.data

    def get_corpus_sparce_rep(self):
        return self.data_sparse

    def apply_transform(self, corpus):
        data = gensim.matutils.corpus2csc(corpus, num_terms=self.data_sparse.shape[0])
        feat = self.tf_idf_transformer.transform(data.T)
        if self.only_presence:
            sklearn.preprocessing.binarize(feat, copy=False)
            # sklearn.preprocessing.Normalizer(copy=False).transform(feat)
        return feat



#####################################

def main():
    test_param = False
    president = True
    read_backup = False

    no_below = 2
    no_above = 10 ** -1
    c = 0.001
    # stemmer = nltk.stem.snowball.FrenchStemmer()
    stemmer = None
    print "Parsing text data..."
    if president:
        FirstParser = ParserCM(r"C:\Users\Usuario\Desktop\ENSTA\M2 UPMC\Cours\FDMS\corpus.tache1.learn-stem.utf8")
        cvs_log = "..\\..\\var_param_CM.csv"
    else:
        FirstParser = ParserCM(r"C:\Users\Usuario\Desktop\ENSTA\M2 UPMC\Cours\FDMS\corpus.movies.learn.utf8")
        cvs_log = "..\\..\\var_param_movies.csv"
    text_data, Y = FirstParser.get_data()
    print "Done parsing\n"

    if test_param:
        count = 0
        with open(cvs_log, "w") as out:
            try:
                for normalization in [None, 'l1', 'l2']:
                    for no_below in 10 ** np.arange(1, 4, 1, dtype=np.float64):
                        for no_above in 10 ** np.arange(-3, 0, 1, dtype=np.float64): #np.arange(0.4, 1.1, 0.1):
                            print "Cleaning parsed text..."
                            if president:
                                Cleaner = GuigueCleaner(text_data, no_below=no_below, no_above=no_above, stoplist=FRStopWords().get_toplist())
                            else:
                                Cleaner = GuigueCleaner(text_data, no_below=no_below, no_above=no_above, stoplist=ENStopWords().get_toplist())
                            cbow_corpus = Cleaner.get_corpus()
                            print "Done cleaning\n"

                            print "Getting features..."
                            tf_idfer = SklearnTfIdf(cbow_corpus, idf=False, only_presence=False, norm=normalization)
                            X = tf_idfer.get_features()
                            print no_below, no_above, X.shape[-1]
                            print "Got\n"

                            print "Learning SVM model..."

                            for kernel in [None]:
                                for c in 10 ** np.arange(-4, 1, dtype=np.float64):
                                    t0 = timeit.default_timer()
                                    # svm = learn_by_svm(X, Y, C=c, kernel=kernel, fit=False, path="..\\..\\svm_dump{}.pkl".format(count))
                                    svm = learn_by_MNbayes(X, Y, alpha=c, fit=False)
                                    scores = svm.cross_val(X, Y, 5, cpus=6)
                                    t1 = timeit.default_timer()
                                    print no_below, no_above, kernel,normalization, c, "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), (t1-t0)/60
                                    out.write("{},{},{},{},{},{}\n".format(no_below, no_above,normalization, kernel, c, scores.mean()))
                                    count += 1
            except KeyboardInterrupt:
                pass

        print "Done\n"
        exit(0)
    else:
        print "Cleaning parsed text..."
        if president:
            Cleaner = GuigueCleaner(text_data, no_below=no_below, no_above=no_above, stoplist=FRStopWords().get_toplist(), stemmer=stemmer)
        else:
            Cleaner = GuigueCleaner(text_data, no_below=no_below, no_above=no_above, stoplist=ENStopWords().get_toplist())
        cbow_corpus = Cleaner.get_corpus()
        print "Done cleaning\n"

        print "Getting features..."
        tf_idfer = SklearnTfIdf(cbow_corpus, idf=True, only_presence=False, norm='l2')
        X = tf_idfer.get_features()
        print X.shape
        print "Got\n"

        kernel = "linear"
        print "Learning SVM model..."
        print datetime.datetime.now().time()
        t0 = timeit.default_timer()
        if os.path.isfile(r"..\..\svm_dump.pkl") and read_backup:
            svm = learn_by_svm()
        else:
            # svm = learn_by_MNbayes(X, Y, alpha=c)
            # svm = learn_by_svm(X, Y, C=c, kernel=kernel)
            svm = learn_by_svc(X, Y, C=c)
            # svm = learn_by_perceptron(X, Y)
        t1 = timeit.default_timer()
        print "Learning took: {} min".format((t1-t0)/60)
        print "Done\n"

    print "Parsing text data..."
    if president:
        TestParser = ParserCMTest(r"C:\Users\Usuario\Desktop\ENSTA\M2 UPMC\Cours\FDMS\corpus.tache1.test-stem.utf8")
        out_file = r"C:\Users\Usuario\Desktop\ENSTA\M2 UPMC\Cours\FDMS\pred_CM"
    else:
        TestParser = ParserCMTest(r"C:\Users\Usuario\Desktop\ENSTA\M2 UPMC\Cours\FDMS\corpus.movies.test.utf8")
        out_file = r"C:\Users\Usuario\Desktop\ENSTA\M2 UPMC\Cours\FDMS\pred_movies"
    test_text = TestParser.get_data()

    print "Cleaning and getting features"
    test_features = tf_idfer.apply_transform(Cleaner.line_list_to_bow(Cleaner.lines_to_words(test_text)))
    print test_features.shape
    # test_features = scaler.transform(test_features)
    print "Prediciton"
    Y_pred = svm.predict(test_features)

    print "Saving outputs"
    with open(out_file, "w") as out:
        for pred in Y_pred:
            if pred <= 0:
                out.write("M\n")
            else:
                out.write("C\n")

    if president:

        mf = MedianFilter(vec=Y_pred)
        mf.export_data(out_file+"_mf")

    print "DONE"
    import winsound
    Freq = 2500 # Set Frequency To 2500 Hertz
    Dur = 1000 # Set Duration To 1000 ms == 1 second
    winsound.Beep(Freq,Dur)

if  __name__ =='__main__':main()