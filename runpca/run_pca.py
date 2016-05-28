#!D:\Miniconda2 python
# encoding: utf-8

"""
@author: MarkLiu
@file: run_pca.py
@time: 2016/5/28 14:28
"""
import numpy as np
from sklearn import decomposition
from dataset import load_save_datas as data


def calcuate_retained_variance(original_data, inverse_transformed_data):
    """
    calcuate retained variance
    :param original_data:
    :param inverse_transformed_data:
    :return:
    """
    diffMat = original_data - inverse_transformed_data
    single_square = np.sum(np.square(diffMat), axis=1)
    squared_projection_error = np.sum(single_square)

    total_variation = np.sum(
        np.sum(np.square(original_data), axis=1)
    )
    error = squared_projection_error / total_variation
    retained_variance = 1 - error

    return retained_variance


def try_n_components(train_data, best_retained_variance, step=0.01):
    """
    尝试不同的 n_components，计算 retained_variances
    :param train_data:
    :param best_retained_variance: 0 ~ 1 ，主成分所占百分比
    :param step: pca_components step
    :return:
    """
    # 待测试的 n_components
    pca_components = np.arange(best_retained_variance * 0.9, 1, step=step)
    # pca_components = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7,
    #                   0.75, 0.8, 0.85, 0.9, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]

    retained_variances = {}
    for comp_size in pca_components:
        print "Fitting pca with %f components" % comp_size
        pca = decomposition.PCA(n_components=comp_size, whiten=True).fit(train_data)
        # Apply the dimensionality reduction on train_data
        transformed_data = pca.transform(train_data)
        # Transform data back to its original space
        inverse_transformed_data = pca.inverse_transform(transformed_data)

        # calcuate retained variance
        retained_variance = calcuate_retained_variance(train_data,
                                                       inverse_transformed_data)
        print "--------> calcuate retained variance: %f" % retained_variance
        retained_variances[comp_size] = retained_variance

    return retained_variances


def choose_best_n_components(train_data, best_retained_variance):
    """
    获取最佳 PCA 参数 n_components
    :param train_data:
    :param best_retained_variance: 0 ~ 1 ，主成分所占百分比
    :return:
    """
    best_n_components = np.Inf
    retained_variances = try_n_components(train_data, best_retained_variance)

    # # 绘制 n_components - retained_variances曲线
    # import matplotlib.pyplot as plt
    # n_components = retained_variances.keys()
    # retained_variances_ = retained_variances.values()
    # plt.scatter(n_components, retained_variances_, c='blue')
    # plt.plot(n_components, retained_variances_, c='red')
    # plt.title(u'n_components - retained_variances曲线')
    # plt.xlabel('n_components')
    # plt.ylabel('retained_variances')
    # plt.savefig('n_components-retained_variances.png')
    # plt.show()

    sorted_retained_variances = sorted(retained_variances.iteritems(),
                                       key=lambda item: item[1],
                                       reverse=True)

    print sorted_retained_variances
    for variances in sorted_retained_variances:
        print variances
        # 获取达到 best_retained_variance 要求的最小的 n_components
        if variances[1] < best_retained_variance:
            break
        best_n_components = variances[0]

    return best_n_components


if __name__ == '__main__':
    print 'loading datas...'
    train_digit_datas, train_digit_labels = data.load_train_data('../dataset/train.csv')
    n_components_ = choose_best_n_components(train_digit_datas, 0.99)
    print 'best n_components:', n_components_
