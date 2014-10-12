#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

__author__ = 'Usuário'

import re
import numpy as np
import codecs
from tools import * # import du fichier défini plus haut
from Cleaner import *


class ParserPresident:
    def __init__(self, data_file, president_dict={'C': 1, "M": 0}, pattern=r"\<([0-9]+):([0-9]+):([a-z]+)\> (.+)"):
        self.in_file = data_file
        self.map_president = president_dict
        self.regex_pattern = re.compile(pattern, re.I | re.MULTILINE)
        with open(self.in_file, 'r') as f:
            # print f.read()
            self.raw_data = f.read()

    #return a three level dictionary [author][doc][line] = text
    def get_matches(self):
        match_results = self.regex_pattern.findall(self.raw_data)

        results = dict()
        for _match in match_results:
            try:
                _author = self.map_president[_match[2]]
            except KeyError:
                print "Map not defined for {}".format(_match[1])
                continue
            try:
                results[_author][_match[0]][_match[1]] = _match[3].decode('utf-8')
            except KeyError:
                try:
                    results[_author][_match[0]] = {_match[1]: _match[3].decode('utf-8')}
                except KeyError:
                    results[_author] = {_match[0]: {_match[1]: _match[3].decode('utf-8')}}

        return results

    def get_vocab(self):
        vocab = dict()

        results_dict = self.get_matches()
        
        for author in results_dict:
            for doc in sorted(results_dict[author].keys(), key=lambda x: int(x)):
                for line in sorted(results_dict[author][doc].keys(), key=lambda x: int(x)):
                    for word in results_dict[author][doc][line].split(" "):
                        try:
                            vocab[word] += 1
                        except KeyError:
                            vocab[word] = 1
        return vocab


class ParserCM:
    def __init__(self, fname):
        self.nlines = compteLignes(fname)

        self.alltxts = []
        self.Y = np.ones(self.nlines )

        with codecs.open(fname, 'r', 'utf-8') as s:
            cpt = 0
            for i in range(self.nlines):
                txt = s.readline()

                lab = re.sub(r"<[0-9]*:[0-9]*:(.)>.*", "\\1", txt)
                txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)", "\\1", txt).strip()

                if lab.count('M') > 0:
                    self.Y[cpt] = -1
                self.alltxts.append(txt)
                cpt += 1

    def get_data(self):
        return self.alltxts, self.Y


class ParserCMTest:
    def __init__(self, fname):
        self.nlines = compteLignes(fname)
        self.alltxts = []
        with codecs.open(fname, 'r', 'utf-8') as s:
            # for i in range(self.nlines):
            #     txt = s.readline()
            #
            #     txt = re.sub(r"<[0-9]*:[0-9]*>(.*)", "\\1", txt).strip()
            #     self.alltxts.append(txt)
            all_text = s.read()
            for txt in all_text.split('\n')[:-1]:
                txt = re.sub(r"<[0-9]*:[0-9]*>(.*)", "\\1", txt).strip()
                self.alltxts.append(txt)

    def get_data(self):
        return self.alltxts


def movies_corpus2president_corpus(a_dir, out, map={'neg': 'C', 'pos': 'M'}, ext='.txt'):
    files = dict()
    for sub_folder in map:
        files[sub_folder] = [a_dir+"/"+sub_folder+"/"+f for f in os.listdir(a_dir+"/"+sub_folder) if f.endswith(ext)]

    with codecs.open(out, 'wb') as out_f:
        for sub_folder in map:
            doc_count = 1
            for _file in files[sub_folder]:
                out_f.write("<{}:1:{}> {}\n".format(doc_count, map[sub_folder], file_text(_file)))
                doc_count += 1


def movies_test2president_test(in_file, out_file):
    with codecs.open(in_file, "rb") as _in:
        with codecs.open(out_file, "wb") as out:
            count = 1
            for line in _in.read().splitlines():
                out.write("<{}:1> {}\n".format(count, line))
                count += 1


def president2stemed(in_file, out_file, stemmer):
    cregex = re.compile(r"(<[0-9]*:[0-9]*:?.?>)(.*)", re.UNICODE)
    comp = re.compile(r'\W+', re.UNICODE)
    count = 0
    with codecs.open(in_file, "r", 'utf-8') as _in:
        with codecs.open(out_file, "w", 'utf-8') as out:
            all_lines = _in.read()
            for line in all_lines.split('\n'):
                tmp = cregex.search(line)
                line_text = tmp.group(2)
                tmp_ = comp.sub(' ', line_text.lower()).strip().split()
                stemed_line = [stemmer.stem(word) for word in tmp_]
                out.write('{} '.format(tmp.group(1)))
                out.write(" ".join(stemed_line))
                out.write('\n')
    print count


def file_text(_file):
    # with codecs.open(_file) as f:
    #     return " ".join(f.read().splitlines())
    return readAFile(_file)


def log2csv(in_, out):
    re_ = re.compile(r'\d* \((\d*), (\d*\.?\d*)\) \((\w*), (\w*), (\'?\w*\'?)\) \((\d*\.?\d*),\) (\d*\.?\d*)')
    with codecs.open(in_, 'r') as fi:
        with codecs.open(out, 'w') as fo:
            for line in fi.read().splitlines():
                mobj = re_.match(line)
                if mobj is None:
                    continue
                print line
                print ','.join([a for a in mobj.groups()])
                fo.write('{}\n'.format(','.join([a for a in mobj.groups()])))


def main():
    # FirstParser = ParserPresident(r"C:\Users\Usuario\Desktop\ENSTA\M2 UPMC\Cours\FDMS\cleaned.corpus.tache1.learn.utf8")
    # tmp = FirstParser.get_matches()
    # vocab = FirstParser.get_vocab()
    # print len(vocab)

    # new_vocab = {k: v for k, v in vocab.items() if v > 10}
    # print len(new_vocab)

    # for word in sorted(new_vocab, key=lambda x: new_vocab[x], reverse=True)[0:20]:
    #     print word, new_vocab[word]

    # FirstParser = ParserPresident(r"C:\Users\Usuario\Desktop\ENSTA\M2 UPMC\Cours\FDMS\cleaned.corpus.tache1.learn.utf8")

    # FirstParser.export_corpus('teste.txt')
    # FirstParser = ParserCM(r"C:\Users\Usuario\Desktop\ENSTA\M2 UPMC\Cours\FDMS\cleaned.corpus.tache1.learn.utf8")
    #
    # Text, Y = FirstParser.get_data()
    #
    # PosCleaner = GuigueCleaner(Text)
    # print len(Text), len(Y)
    # print Text[0], Y[0]
    # movies_corpus2president_corpus(r"C:\Users\Usuario\Desktop\ENSTA\M2 UPMC\Cours\FDMS\movies1000", r"..\..\corpus.movies.learn.utf8")
    # movies_test2president_test(r"C:\Users\Usuario\Desktop\ENSTA\M2 UPMC\Cours\FDMS\movies1000\testSentiment.txt", r"..\..\corpus.movies.test.utf8")
    # president2stemed(r"..\..\corpus.movies.learn.utf8", r"..\..\corpus.movies.learn-stem.utf8", nltk.stem.snowball.EnglishStemmer())
    # president2stemed(r"..\..\corpus.movies.test.utf8", r"..\..\corpus.movies.test-stem.utf8", nltk.stem.snowball.EnglishStemmer())
    #
    # president2stemed(r"..\..\corpus.tache1.learn.utf8", r"..\..\corpus.tache1.learn-stem.utf8", nltk.stem.snowball.FrenchStemmer())
    # president2stemed(r"..\..\corpus.tache1.test.utf8", r"..\..\corpus.tache1.test-stem.utf8", nltk.stem.snowball.FrenchStemmer())
    log2csv(r'./log/cm-svm-ns.log', '../../cm-svm-notstemed.csv')
    log2csv(r'./log/cm-svm-s.log', '../../cm-svm-stemed.csv')

if  __name__ =='__main__':main()
