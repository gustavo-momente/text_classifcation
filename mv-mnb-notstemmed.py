#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Usuário'

import itertools
import mechanize
import re
from Parser import *
from Filter import *
from Cleaner import *
from Learners import *
from stopwords import *
from Features import *
from query_server import query_server
import time


def main():
    president = False

    # model_type = "MNB"
    model_type = "110"
    print "Parsing text data..."

    FirstParser = ParserCM(r"C:\Users\Usuario\Desktop\ENSTA\M2 UPMC\Cours\FDMS\corpus.movies.learn.utf8")
    cvs_log = "..\\..\\mv-mnb-notstemed.csv"
    stemmer = None
    text_data, Y = FirstParser.get_data()

    print "Parsing test data..."
    TestParser = ParserCMTest(r"C:\Users\Usuario\Desktop\ENSTA\M2 UPMC\Cours\FDMS\corpus.movies.test.utf8")
    out_file = r"C:\Users\Usuario\Desktop\ENSTA\M2 UPMC\Cours\FDMS\110pred_movies"
    up_file = out_file
    task = "sentiment.txt"


    test_text = TestParser.get_data()
    print "Done parsing\n"


    count = 0
    mod_folder = "../../"+model_type

    mod_folder += "_mv"


    try:
        os.mkdir(mod_folder)
    except WindowsError:
        pass
    cleaner_hyper_name = "no_below,no_above"
    feature_name = 'idf,bool,norm'
    parameter_name = 'alpha'

    with open(cvs_log, "w") as out_log:
        try:
            out_log.write("{},{},{},{}\n".format(cleaner_hyper_name, feature_name, parameter_name, "score"))
            # FIRST FOR

            cleaner_iter = itertools.product([2, 4, 5, 10, 20], np.arange(.1, 0.9, 0.2, dtype=np.float64))
            for cleaner_hyper in cleaner_iter:
                no_below, no_above = cleaner_hyper


                print "Cleaning parsed text..."
                Cleaner = GuigueCleaner(text_data, no_below=no_below, no_above=no_above, stoplist=FRStopWords().get_toplist())
                cbow_corpus = Cleaner.get_corpus()
                print "Done cleaning\n"

                # SECOND FOR use a tuple!
                print "Getting features..."
                feature_config = [feature for feature in itertools.product([True, False], [False], [None, 'l2'])]
                feature_config.append((False, True, None))

                for feature_type in feature_config:
                    tf_idfer = SklearnTfIdf(cbow_corpus, idf=feature_type[0], only_presence=feature_type[1], norm=feature_type[2])
                    X = tf_idfer.get_features()
                    print X.shape
                    if X.shape[-1] == 0:
                        continue

                    print "Got\n"

                    param_config = itertools.product(10 ** np.arange(-3, 2, 1, dtype=np.float64))
                    for param in param_config:

                        print "Learning model..."
                        print datetime.datetime.now().time()
                        t0 = timeit.default_timer()
                        try:
                            os.mkdir("{}/{}".format(mod_folder, count))
                        except WindowsError:
                            pass
                        svm = learn_by_MNbayes(X, Y, alpha=param[0], path="{}/{}/MNB_dump.pkl".format(mod_folder, count))
                        t1 = timeit.default_timer()
                        print "Learning took: {} min".format((t1-t0)/60)
                        print "Done\n"


                        print "Cleaning and getting features"
                        test_features = tf_idfer.apply_transform(Cleaner.line_list_to_bow(Cleaner.lines_to_words(test_text)))
                        print test_features.shape

                        print "Prediciton"
                        Y_pred = svm.predict(test_features)

                        print "Saving outputs"
                        with open(out_file, "w") as out:
                            for pred in Y_pred:
                                if pred <= 0:
                                    out.write("M\n")
                                else:
                                    out.write("C\n")

                        score = query_server(up_file, task)
                        out_log.write("{},{},{},{}\n".format(','.join(['{}'.format(x) for x in cleaner_hyper]),
                                                             ','.join(['{}'.format(x) for x in feature_type]),
                                                             ','.join(['{}'.format(x) for x in param]), score))
                        count += 1
                        print count, cleaner_hyper,\
                            feature_type, param, query_server(up_file, task)
                        time.sleep(1)
        except KeyboardInterrupt:
            pass
    print "DONE"
    import winsound
    Freq = 2500 # Set Frequency To 2500 Hertz
    Dur = 1000 # Set Duration To 1000 ms == 1 second
    winsound.Beep(Freq,Dur)


if  __name__ =='__main__':main()