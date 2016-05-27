#!D:\Miniconda2 python
# encoding: utf-8

"""
@author: MarkLiu
@file: plot_digit.py
@time: 2016/5/27 14:47
"""
import numpy as np
from matplotlib.pylab import plt


def plot_count_digit_img(digit_datas, count=100):
    """
    绘制100个数据
    :return:
    """
    digit_datas = np.matrix(digit_datas)
    random_index = np.arange(0, count)
    for i in random_index:
        plt.subplot(10, 10, i + 1)  # 假设100个
        plt.subplots_adjust(wspace=0.1, hspace=0.1)  # 调整subplot的间隔
        data = digit_datas[i].reshape((28, 28))
        plt.imshow(data, cmap=plt.get_cmap('gray'))
        plt.xticks([])
        plt.yticks([])
    plt.show()
