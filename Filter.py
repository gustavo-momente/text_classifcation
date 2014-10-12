#!/usr/bin/env python
# -*- coding: utf-8 -*-


__author__ = 'Usuário'
import scipy.signal as ft
import numpy as np


class MedianFilter:
    def __init__(self, vec=None, path=None, out_dict={'M': -1, 'C': 1}):
        self.inv_map = {v: k for k, v in out_dict.items()}
        self.vec = []
        if vec is None:
            with open(path, 'r') as f:
                for line in f.read().splitlines():
                    self.vec.append(out_dict[line])
        else:
            self.vec = vec
        #self.filtered = np.floor(ft.medfilt(self.vec, kernel_size=3)).astype(int)

        k = 5
        self.x = np.array(self.vec)
        k2 = (k - 1) // 2
        self.y = np.zeros((len(self.x), k), dtype=self.x.dtype)

        update = 1
        count = 0
        while update != 0:

            self.y[:, k2] = self.x
            for i in range (k2):
                j = k2 - i
                self.y[j:, i] = self.x[:-j]
                self.y[:j, i] = self.x[0]
                self.y[:-j, -(i+1)] = self.x[j:]
                self.y[-j:, -(i+1)] = self.x[-1]
            self.filtered = np.median(self.y, axis=1)
            update = np.linalg.norm(self.filtered - self.x, ord=1)
            self.x = self.filtered
            count += 1
        print "Filter iterations: ", count

    def export_data(self, path):
        with open(path, 'w') as f:
            for i in xrange(len(self.vec)):
                val = self.filtered[i]
                f.write("{}\n".format(self.inv_map[val]))


def main():
    myFilter = MedianFilter(path=r"C:\Users\Usuario\Desktop\ENSTA\M2 UPMC\Cours\FDMS\pred1")
    myFilter.export_data(r"..\..\pred1_filtered")



if  __name__ =='__main__':main()