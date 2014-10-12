#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools

__author__ = 'Usuário'

import mechanize
import re
from Parser import *
from Filter import *
from Cleaner import *
from Learners import *
from stopwords import *
from Features import *
import time
import urllib2

def query_server(FILENAME=r"C:\Users\Usuario\Desktop\ENSTA\M2 UPMC\Cours\FDMS\pred_CM_mf", task="chiracmitterand.txt"):
    br = mechanize.Browser()
    # br.set_all_readonly(False)  # allow everything to be written to
    br.set_handle_robots(False)  # ignore robots
    br.set_handle_refresh(False)  # can sometimes hang without this

    url = r'http://project.lip6.fr:8180/servlets/ScoringMAIAD'
    tries = 0
    while True:
        try:
            response = br.open(url)
            break
        except urllib2.URLError:
            tries += 1
            if tries > 5:
                return -2
            time.sleep(60)


    br.form = list(br.forms())[0]  # use when form is unnamed

    Login = br.form.find_control("Login")
    Login.value = "VILELA"
    Password = br.form.find_control("Password")
    Password.value = "Gustavo"
    br.form.add_file(open(FILENAME), 'text/plain', FILENAME)
    Task = br.form.find_control("Task")
    Task.value = [task]

    tries = 0
    while True:
        try:
            response = br.submit()
            text = response.read()

            pattern = re.compile(r"<BR>Score: ([0-9]+\.[0-9]*)")

            matches = pattern.search(text)
            score = float(matches.group(1))
            break

        except:
            tries += 1
            if tries > 5:
                score = -1
                break
            time.sleep(60)

    # text = response.read()
    #
    # pattern = re.compile(r"<BR>Score: ([0-9]+\.[0-9]*)")
    #
    # matches = pattern.search(text)
    # score = float(matches.group(1))
    return score


def main():
    test_param = True
    president = False
    read_backup = False

    # model_type = "MNB"
    model_type = "SVC"
    print "Parsing text data..."
    if president:
        FirstParser = ParserCM(r"C:\Users\Usuario\Desktop\ENSTA\M2 UPMC\Cours\FDMS\corpus.tache1.learn.utf8")
        cvs_log = "..\\..\\var_param_CM.csv"
        # stemmer = nltk.stem.snowball.FrenchStemmer()
        stemmer = None
    else:
        FirstParser = ParserCM(r"C:\Users\Usuario\Desktop\ENSTA\M2 UPMC\Cours\FDMS\corpus.movies.learn.utf8")
        cvs_log = "..\\..\\var_param_movies.csv"
        # stemmer = nltk.stem.snowball.EnglishStemmer()
        stemmer = None
    text_data, Y = FirstParser.get_data()

    print "Parsing test data..."
    if president:
        TestParser = ParserCMTest(r"C:\Users\Usuario\Desktop\ENSTA\M2 UPMC\Cours\FDMS\corpus.tache1.test.utf8")
        out_file = r"C:\Users\Usuario\Desktop\ENSTA\M2 UPMC\Cours\FDMS\pred_CM"
        up_file = out_file+"_mf"
        task = "chiracmitterand.txt"
    else:
        TestParser = ParserCMTest(r"C:\Users\Usuario\Desktop\ENSTA\M2 UPMC\Cours\FDMS\corpus.movies.test.utf8")
        out_file = r"C:\Users\Usuario\Desktop\ENSTA\M2 UPMC\Cours\FDMS\pred_movies"
        up_file = out_file
        task = "sentiment.txt"

    test_text = TestParser.get_data()
    print "Done parsing\n"


    count = 0
    mod_folder = "../../"+model_type
    if president:
        mod_folder += "_cm"
    else:
        mod_folder += "_mov"

    try:
        os.mkdir(mod_folder)
    except WindowsError:
        pass
    cleaner_hyper_name = "no_below,no_above"
    feature_name = 'idf,bool,norm'
    if model_type == 'MNB':
        parameter_name = 'alpha'
    else:
        parameter_name = 'C'
    with open(cvs_log, "w") as out_log:
        try:
            out_log.write("{},{},{},{}\n".format(cleaner_hyper_name, feature_name, parameter_name, "score"))
            # FIRST FOR
            if model_type == 'MNB':
                cleaner_iter = itertools.product([2, 3, 4, 5], 10 ** np.arange(-2, 0, .5, dtype=np.float64))
            else:
                cleaner_iter = itertools.product([2, 3, 4, 5, 10], 10 ** np.arange(-2, 0, .5, dtype=np.float64))
            for cleaner_hyper in cleaner_iter:
                no_below, no_above = cleaner_hyper


                print "Cleaning parsed text..."
                if president:
                    Cleaner = GuigueCleaner(text_data, no_below=no_below, no_above=no_above, stoplist=FRStopWords().get_toplist())
                else:
                    Cleaner = GuigueCleaner(text_data, no_below=no_below, no_above=no_above, stoplist=ENStopWords().get_toplist(), stemmer=stemmer)
                cbow_corpus = Cleaner.get_corpus()
                print "Done cleaning\n"

                # SECOND FOR use a tuple!
                print "Getting features..."
                if model_type == 'MNB':
                    feature_config = [feature for feature in itertools.product([True, False], [False], [None])]
                    feature_config.append((False, True, None))
                else:
                    # feature_config = [feature for feature in itertools.product([True, False], [False], [None, 'l2'])]
                    # feature_config.append((False, True, None))
                    feature_config = [feature for feature in itertools.product([True], [False], [None])]

                for feature_type in feature_config:
                    tf_idfer = SklearnTfIdf(cbow_corpus, idf=feature_type[0], only_presence=feature_type[1], norm=feature_type[2])
                    X = tf_idfer.get_features()
                    print X.shape
                    if X.shape[-1] == 0:
                        continue

                    print "Got\n"
                    if model_type == 'MNB':
                        param_config = itertools.product(10 ** np.arange(-3, 1, 1, dtype=np.float64))
                    else:
                        param_config = itertools.product(10 ** np.arange(-3, 2, 1, dtype=np.float64))
                    for param in param_config:

                        print "Learning model..."
                        print datetime.datetime.now().time()
                        t0 = timeit.default_timer()
                        try:
                            os.mkdir("{}/{}".format(mod_folder, count))
                        except WindowsError:
                            pass
                        if model_type == 'MNB':
                            svm = learn_by_MNbayes(X, Y, alpha=param[0], path="{}/{}/MNB_dump.pkl".format(mod_folder, count))
                        else:
                            svm = learn_by_svc(X, Y, C=param[0],  path="{}/{}/SVC_dump.pkl".format(mod_folder, count))
                        t1 = timeit.default_timer()
                        print "Learning took: {} min".format((t1-t0)/60)
                        print "Done\n"


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

if __name__ == '__main__': main()
