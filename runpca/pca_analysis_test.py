#!D:\Miniconda2 python
# encoding: utf-8

"""
@author: MarkLiu
@file: runpca.py
@time: 2016/5/27 21:36
"""
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn import decomposition
import csv


def read_data(filname, limit=None):
    data = []
    labels = []

    csv_reader = csv.reader(open(filname, "r"), delimiter=",")
    index = 0
    for row in csv_reader:
        index += 1
        if index == 1:
            continue

        labels.append(int(row[0]))
        row = row[1:]

        data.append(np.array(np.int64(row)))

        if limit != None and index == limit + 1:
            break

    return data, labels


print "Reading train data"
train, target = read_data("../dataset/train.csv")

pca_components = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
pca_fits = []

for comp_size in pca_components:
    print "Fitting pca with %f components" % comp_size
    pca_fits.append(decomposition.PCA(n_components=comp_size).fit(train))

figure = plt.figure()

t = np.array(target)

choosen_numbers = []

print 'np.argwhere(t == 0):'
print np.argwhere(t == 0)
choosen_numbers.append(np.argwhere(t == 0)[0])
choosen_numbers.append(np.argwhere(t == 1)[0])
choosen_numbers.append(np.argwhere(t == 2)[0])
choosen_numbers.append(np.argwhere(t == 3)[0])
choosen_numbers.append(np.argwhere(t == 4)[0])
choosen_numbers.append(np.argwhere(t == 5)[0])
choosen_numbers.append(np.argwhere(t == 6)[0])
choosen_numbers.append(np.argwhere(t == 7)[0])
choosen_numbers.append(np.argwhere(t == 8)[0])
choosen_numbers.append(np.argwhere(t == 9)[0])

pca_index = 1
for n in choosen_numbers:
    for p in pca_fits:
        transformed = p.transform(train[n])
        # print "Shape of transformed: %d" % transformed.shape
        reconstructed = p.inverse_transform(transformed)
        f = figure.add_subplot(10, len(pca_components), pca_index).imshow(np.reshape(reconstructed, (28, 28)),
                                                                          interpolation='nearest',
                                                                          cmap=cm.hot)  # cmap=cm.Greys_r)
        for xlabel_i in f.axes.get_xticklabels():
            xlabel_i.set_visible(False)
            xlabel_i.set_fontsize(0.0)
        for xlabel_i in f.axes.get_yticklabels():
            xlabel_i.set_fontsize(0.0)
            xlabel_i.set_visible(False)
        for tick in f.axes.get_xticklines():
            tick.set_visible(False)
        for tick in f.axes.get_yticklines():
            tick.set_visible(False)
        pca_index += 1

plt.show()
