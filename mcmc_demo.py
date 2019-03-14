# -*- coding: utf-8 -*-
import numpy as np
import seaborn as sns
import pandas as pd


def fun(x):
    return x**2


def test_uniform_mc(a, b, size=100000):
    total_num = size
    unx = np.random.uniform(a, b, total_num)
    unx_y = [fun(x) for x in unx]
    total_sum = np.sum(unx_y)
    print(total_sum/total_num*(b-a))


def test_markov_stock():
    matrix = np.matrix([[0.9, 0.075, 0.025], [0.15, 0.8, 0.05], [
                       0.25, 0.25, 0.5]], dtype=float)
    v1 = np.matrix([[0.3, 0.4, 0.3]], dtype=float)
    for i in range(100):
        v1 *= matrix
        print(i, v1)


def test_bivar_gaussian_gibbs(mu, sigma, iter=int(5e3)):
    def gen_y_given_x(x, mus=mu, sigmas=sigma):
        mu = mus[1] + sigmas[0, 1] / sigmas[1, 1] * (x - mus[0])
        sigma = sigmas[1, 1] - sigmas[0, 1] / sigmas[0, 0] * sigmas[0, 1]
        return np.random.normal(mu, sigma)

    def gen_x_given_y(y, mus=mu, sigmas=sigma):
        mu = mus[0] + sigmas[1, 0] / sigmas[0, 0] * (y - mus[1])
        sigma = sigmas[0, 0] - sigmas[1, 0] / sigmas[1, 1] * sigmas[1, 0]
        return np.random.normal(mu, sigma)

    # print(mu.shape)
    var_num = mu.shape[0]

    x = np.random.normal(0.5, 1)
    samples = np.zeros((iter, var_num))

    for i in range(iter):
        y = gen_y_given_x(x)
        x = gen_x_given_y(y)
        samples[i, :] = [x, y]

    return samples


def plot_samples(x, y, data=None, picName='out'):
    samples = pd.DataFrame(data=data, columns=[x, y])
    sns_plot = sns.jointplot(x, y, data=samples)
    sns_plot.savefig('./pic/{}.png'.format(picName))


def test_normal_distribuiton(mu=0, sigma=1):
    samples = np.random.normal(mu, sigma, int(1e6))
    sns_plot = sns.distplot(samples)
    fig = sns_plot.get_figure()
    fig.savefig('./pic/normal_dist.png')


if __name__ == "__main__":
    # test_uniform_mc(0, 3)
    # test_markov_stock()

    # test_bivar_gaussian_gibbs
    # mus = np.array([0.5, 0.5])
    # sigmas = np.array([[1, 0], [0, 1]])

    # # plot the ground truth
    # samples = np.random.multivariate_normal(mus, sigmas, int(1e6))
    # plot_samples('x', 'y', samples, 'out1')
    # # plot the Gibbs sampling results
    # samples = test_bivar_gaussian_gibbs(mus, sigmas, iter=int(1e6))
    # plot_samples('x', 'y', samples, 'out2')

    # test_normal_distribution()
    test_normal_distribuiton()
