#!/home/sunnymarkliu/software/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 16-10-25 下午8:42
"""
import pandas
import numpy as np
from scipy import stats


def load_result_csv(filename):
    result = pandas.read_csv(filename)
    predict_labels = []
    for index, row in result.iterrows():
        key = row['Label']
        predict_labels.append(key)

    return predict_labels


result = []
result1 = load_result_csv('tf_cnn_test_labels_1.csv')
result2 = load_result_csv('tf_cnn_test_labels_2.csv')
result3 = load_result_csv('tf_cnn_test_labels_3.csv')
result4 = load_result_csv('tf_cnn_test_labels_4.csv')

result.append(result1)
result.append(result2)
result.append(result3)
result.append(result4)
result = np.mat(result)

print 'calcuate the mode...'
result_mode = stats.mode(result, axis=0)
predict_labels = result_mode[0][0]
predict_labels = predict_labels.reshape(len(result1), 1)
predict_labels = np.append([100], predict_labels)
df = pandas.DataFrame(predict_labels)
df.to_csv('tf_cnn_test_labels.csv', sep=',')
print 'done.'
