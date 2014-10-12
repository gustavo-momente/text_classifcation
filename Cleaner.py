#!/usr/bin/env python
# -*- coding: utf-8 -*-



__author__ = 'Usuário'

import re
from Parser import *
import codecs
import nltk
from gensim import corpora

class LineCleaner:
    def __init__(self, patterns=[r'\W+'], replaces=[' ']):
        self.pr = patterns
        self.rp = replaces
        self.compiled_pr = list()

        for pattern in self.pr:
            self.compiled_pr.append(re.compile(pattern, re.UNICODE))

    def clean(self, in_line):
        line = in_line
        for regex_pos in range(len(self.pr)):
            line = self.compiled_pr[regex_pos].sub(self.rp[regex_pos], line).strip()
        return line


class NaiveCleaner:
    def __init__(self, data_file, president_dict={'C': 1, "M": 0}, pattern=r"\<([0-9]+):([0-9]+):([a-z]+)\> (.+)"):
        self.raw_parser = ParserPresident(data_file, president_dict=president_dict, pattern=pattern)
        self.parsed_text = self.raw_parser.get_matches()
        self.lc = LineCleaner()
        self.reverse_pd = {v: k for k, v in president_dict.items()}
        self.is_clean = False

    def text_cleaner(self):
        for author in self.parsed_text:
            for doc in self.parsed_text[author].keys():
                for line in self.parsed_text[author][doc]:
                    self.parsed_text[author][doc][line] = self.lc.clean(self.parsed_text[author][doc][line])
        self.is_clean = True

    def export_data(self, out_file):
        if not self.is_clean:
            self.text_cleaner()

        with codecs.open(out_file, 'w', 'utf-8') as f:
            for author in self.parsed_text:
                for doc in sorted(self.parsed_text[author].keys(), key=lambda x: int(x)):
                    for line in sorted(self.parsed_text[author][doc].keys(), key=lambda x: int(x)):
                        f.write(u"<{}:{}:{}> {}\n".format(doc, line, self.reverse_pd[author],
                                                          self.parsed_text[author][doc][line]))


class GuigueCleaner:
    def __init__(self, text_data, no_below=10, no_above=0.8, splitters=u'; |, |\*|\.| |\'|\(|\)|\[|\]|',
                 stoplist=set('le la les de des à un une en au ne ce d l c s je tu il que qui mais quand'.split()),
                 stemmer=None):
        stoplist.add('')
        self.stoplist = stoplist
        self.splitters = splitters
        self.stemmer = stemmer
        # self.dictionary = corpora.Dictionary(re.split(self.splitters, doc.lower().strip()) for doc in text_data)
        self.comp = re.compile(r'\W+', re.UNICODE)
        texts = [self.comp.sub(' ', doc.lower()).strip().split() for doc in text_data]
        if self.stemmer is not None:
            texts = [[self.stemmer.stem(word) for word in line] for line in texts]
        self.dictionary = corpora.Dictionary(texts)

        stop_ids = [self.dictionary.token2id[stopword] for stopword in stoplist if stopword in self.dictionary.token2id]
        # once_ids = [self.tokenid for tokenid, docfreq in self.dictionary.dfs.iteritems() if docfreq < 10 ]
        self.dictionary.filter_tokens(bad_ids=stop_ids) # remove stop words and words that appear only once
        self.dictionary.compactify() # remove gaps in id sequence after words that were removed

        self.dictionary.filter_extremes(no_below=no_below, no_above=no_above)
        # self.texts = [[word for word in re.split(self.splitters, document.lower()) if word not in self.stoplist]
        #               for document in text_data]
        self.corpus = [self.dictionary.doc2bow(text) for text in texts]

    def get_dictionary(self):
        return self.dictionary

    def get_corpus(self):
        return self.corpus

    def lines_to_words(self, lines):
        # return [[word for word in re.split(self.splitters, document.lower().strip()) if word not in self.stoplist] for document
        #         in lines]
        tmp = [self.comp.sub(' ', doc.lower()).strip().split() for doc in lines]
        if self.stemmer is not None:
            tmp = [[self.stemmer.stem(word) for word in line] for line in tmp]
        return tmp

    def get_bow(self, line):
        return self.dictionary.doc2bow(line)

    def line_list_to_bow(self, line_list):
        return [self.dictionary.doc2bow(text) for text in line_list]

def main():
    cleaner = NaiveCleaner(r"C:\Users\Usuario\Desktop\ENSTA\M2 UPMC\Cours\FDMS\corpus.tache1.learn.utf8")
    cleaner.export_data("../../cleaned.corpus.tache1.learn.utf8")


if  __name__ =='__main__':main()