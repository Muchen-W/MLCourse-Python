from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

import copy
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm


def read_file(fr='ex1data1.txt'):
    fp = open(fr, 'r')
    lines = fp.read()
    lines = lines.split()
    xx = []
    yy = []
    for line in lines:
        x, y = line.split(sep=',')
        xx.append([float(x)])
        yy.append([float(y)])
    scatter_plot(xx, yy)
    return conv_matrix(xx, yy)


def scatter_plot(xx, yy):
    plt.figure()
    plt.scatter(xx, yy, marker='x', c='r')
    #plt.show()


def conv_matrix(xx, yy):
    X = np.array([[1] + x for x in xx])
    yy = np.array(yy)
    theta = [0, 0]
    #print(theta, type(theta), len(theta))
    computeCost(X, yy, theta)
    return X, yy


def computeCost(X, yy, theta):
    m = len(yy)
    J_theta = 0
    for i in range(m):
        J_theta += (X[i].dot(theta) - yy[i])**2
    J_theta *= 1/(2*m)
#    print(J_theta)
    return J_theta


def gradientDescent(X, yy, theta=[0, 0], alpha=0.01, iterations=1500):
    m = len(yy)
    for iter_count in range(iterations):
        computeCost(X, yy, theta)
        theta_tmp = list(theta)         # theta_tmp = copy.copy(theta)
        for j in range(len(theta)):
            theta[j] -= alpha/m*sum([(X[i].dot(theta_tmp) - yy[i])*X[i][j] for i in range(m)])
    print(theta)
    plot_regression(X, theta)
    return theta


def plot_regression(X, theta):
    xx = list(zip(*X))
    plt.plot(xx[1], X.dot(theta))
    # plt.show()


def vis_cost(X, yy):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    theta0 = np.arange(-10, 10, 0.2)
    theta1 = np.arange(-1, 4, 0.05)
    # J_cost = np.zeros((101, 101))
    # for i in theta0:
    #     for j in theta1:
    #         theta = [theta0[i], theta1[j]]
    #         J_cost[i][j] = computeCost(X, yy, theta)
    theta0, theta1 = np.meshgrid(theta0, theta1)
    # print([th0 for th0, th1 in zip(theta0, theta1)])
    J_cost = np.array([computeCost(X, yy, [th0, th1]) for th0, th1 in zip(theta0, theta1)])

    ax.plot_surface(theta0, theta1, J_cost)
    # plt.show()
    # print(J_cost)


def main():
    X, yy = read_file()
    gradientDescent(X, yy)
    vis_cost(X, yy)
    plt.show()


if __name__ == '__main__':
    main()
