from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

import copy
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm


def read_file(fr='ex1\ex1data1.txt'):
    fp = open(fr, 'r')
    lines = fp.read()
    lines = lines.split()
    xx = []
    yy = []
    for line in lines:
        x = line.split(sep=',')[:-1]
        y = line.split(sep=',')[-1]
        xx.append([float(item) for item in x])
        yy.append([float(y)])
    if len(xx[0]) == 1:
        scatter_plot(xx, yy)
    elif len(xx[0]) == 2:
        scatter_plot3d(xx, yy)
    fp.close()
    return conv_matrix(xx, yy)


def scatter_plot(xx, yy):
    plt.figure()
    plt.scatter(xx, yy, marker='x', c='r')
    #plt.show()


def scatter_plot3d(xx, yy):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    xs, ys = zip(*xx)
    zs = yy
    ax.scatter(xs, ys, zs, marker='o')
    #plt.show()


def conv_matrix(xx, yy):
    xx = norm_feature(xx)
    X = [[1] + x for x in xx]
    theta = [0] * len(X[0])
    #print(theta, type(theta), len(theta))
    computeCost(X, yy, theta)
    return X, yy


def norm_feature(xx):
    features = [list(x) for x in zip(*xx)]
    for feature in features:
        avg = np.mean(feature)
        std = np.std(feature)
        for i in range(len(feature)):
            feature[i] = (feature[i] - avg) / std
    xx = [list(x) for x in zip(*features)]
    return xx


def computeCost(X, yy, theta):
    m = len(yy)
    J_theta = 0
    for i in range(m):
        J_theta += (np.dot(X[i], theta) - yy[i])**2
    J_theta *= 1/(2*m)
    # print(J_theta, theta)
    return J_theta


def gradientDescent(X, yy, alpha=0.1, iterations=50):
    m = len(yy)
    theta = [0] * len(X[0])
    J_theta = []
    for iter_count in range(iterations):
        J_theta.append(computeCost(X, yy, theta))
        theta_tmp = list(theta)         # theta_tmp = copy.copy(theta)
        for j in range(len(theta)):
            theta[j] -= alpha/m*sum([(np.dot(X[i], theta_tmp) - yy[i])*X[i][j] for i in range(m)])
    plt.figure()
    plt.plot(range(iterations), J_theta)
    print(theta)
    return theta


def main():
    #X, yy = read_file()
    X, yy = read_file(fr='ex1\ex1data2.txt')
    gradientDescent(X, yy)
    plt.show()
    #print(X)
    #print(yy)


if __name__ == '__main__':
    main()
