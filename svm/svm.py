#!D:\Miniconda2 python
# encoding: utf-8

"""
@author: MarkLiu
Average Accuracy: 0.98271
@file: svm.py
@time: 2016/5/27 21:19
"""
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import svm

# The competition datafiles are in the directory ../input
# Read competition data files:
print 'loading datas...'
train = pd.read_csv("../dataset/train.csv")
test = pd.read_csv("../dataset/test.csv")

train_x = train.values[:, 1:]
train_y = train.ix[:, 0]
test_x = test.values

print 'load datas done.'
pca = PCA(n_components=0.9, whiten=True)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)

print 'PCA done.'
svc = svm.SVC(kernel='rbf', C=3)
svc.fit(train_x, train_y)

print 'SVM training done.'
print 'SVM predict...'
test_y = svc.predict(test_x)
pd.DataFrame({"ImageId": range(1, len(test_y) + 1), "Label": test_y}).to_csv('out.csv', index=False, header=True)
