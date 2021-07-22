# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 20:11:16 2020

@author: Amina Asif
"""

from numpy.random import randn, randint  # importing randn
import numpy as np  # importing numpy
import matplotlib.pyplot as plt  # importing plotting module
import itertools
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.stats import kde


def plotDensity_2d(X, Y):
    nbins = 200
    minx, maxx = np.min(X[:, 0]), np.max(X[:, 0])
    miny, maxy = np.min(X[:, 1]), np.max(X[:, 1])
    xi, yi = np.mgrid[minx:maxx:nbins * 1j, miny:maxy:nbins * 1j]

    def calcDensity(xx):
        k = kde.gaussian_kde(xx.T)
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        return zi.reshape(xi.shape)

    pz = calcDensity(X[Y == 1, :])
    nz = calcDensity(X[Y == -1, :])

    c1 = plt.contour(xi, yi, pz, cmap=plt.cm.Greys_r, levels=np.percentile(pz, [75, 90, 95, 97, 99]));
    plt.clabel(c1, inline=1)
    c2 = plt.contour(xi, yi, nz, cmap=plt.cm.Purples_r, levels=np.percentile(nz, [75, 90, 95, 97, 99]));
    plt.clabel(c2, inline=1)
    plt.pcolormesh(xi, yi, 1 - pz * nz, cmap=plt.cm.Blues, vmax=1, vmin=0.99);
    plt.colorbar()
    markers = ('s', 'o')
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], marker=markers[0], c='y', s=30)
    plt.scatter(X[Y == -1, 0], X[Y == -1, 1], marker=markers[1], c='c', s=30)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    #
    plt.grid()
    plt.show()


def plotit(X, Y=None, clf=None, markers=('s', 'o'), hold=False, transform=None):
    """
    Just a function for showing a data scatter plot and classification boundary
    of a classifier clf
    """
    eps = 1e-6
    minx, maxx = np.min(X[:, 0]), np.max(X[:, 0])
    miny, maxy = np.min(X[:, 1]), np.max(X[:, 1])

    if clf is not None:
        npts = 150
        x = np.linspace(minx, maxx, npts)
        y = np.linspace(miny, maxy, npts)
        t = np.array(list(itertools.product(x, y)))
        if transform is not None:
            t = transform(t)
        z = clf(t)
        z = np.reshape(z, (npts, npts)).T
        extent = [minx, maxx, miny, maxy]
        plt.contour(x, y, z, [-1 + eps, 0, 1 - eps], linewidths=[2], colors=('b', 'k', 'r'), extent=extent)
        # plt.imshow(np.flipud(z), extent = extent, cmap=plt.cm.Purples, vmin = -2, vmax = +2); plt.colorbar()
        plt.pcolormesh(x, y, z, cmap=plt.cm.Purples, vmin=-2, vmax=+2);
        plt.colorbar()
        plt.axis([minx, maxx, miny, maxy])

    if Y is not None:

        plt.scatter(X[Y == 1, 0], X[Y == 1, 1], marker=markers[0], c='y', s=30)
        plt.scatter(X[Y == -1, 0], X[Y == -1, 1], marker=markers[1], c='c', s=30)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')

    else:
        plt.scatter(X[:, 0], X[:, 1], marker='.', c='k', s=5)
    if not hold:
        plt.grid()

        plt.show()


def accuracy(ytarget, ypredicted):
    return np.sum(ytarget == ypredicted) / len(ytarget)


class NN:
    def __init__(self, k=1):
        self.K = k

    def fit(self, X, Y):
        self.Xtr = X
        self.Ytr = Y

    def predict(self, Xts):
        Yts = []
        Xts = np.array(Xts)
        for t in Xts:
            distances = np.sqrt(np.sum(np.power((self.Xtr - t), 2), axis=1))
            k_nearest = self.Ytr[np.argsort(distances)[:self.K]]
            y = np.sign(np.sum(k_nearest))
            Yts.append(y)
        return Yts


def getExamples(n=100, d=2):
    """
    Generates n d-dimensional normally distributed examples of each class        
    The mean of the positive class is [1] and for the negative class it is [-1]
    DO NOT CHANGE THIS FUNCTION
    """
    Xp = randn(n, d) + 1  # generate n examples of the positie class
    # Xp[:,0]=Xp[:,0]+1
    Xn = randn(n, d) - 1  # generate n examples of the negative class
    # Xn[:,0]=Xn[:,0]-1
    X = np.vstack((Xp, Xn))  # Stack the examples together to a single matrix
    Y = np.array([+1] * n + [-1] * n)  # Associate Labels
    return X, Y


if __name__ == '__main__':
    # Data Generation and Density Plotting
    n = 100  # number of examples of each class
    d = 2  # number of dimensions
    K = int(input("Enter the value of K:"))
    Xtr, Ytr = getExamples(n=n, d=d)  # Generate Training Examples
    print("Number of positive examples in training: ", np.sum(Ytr == 1))
    print("Number of negative examples in training: ", np.sum(Ytr == -1))
    print("Dimensions of the data: ", Xtr.shape[1])
    Xtt, Ytt = getExamples(n=n, d=d)  # Generate Testing Examples
    plt.figure()
    plt.title("Train Data")
    plotDensity_2d(Xtr, Ytr)

    plt.title("Test Data")
    plotDensity_2d(Xtt, Ytt)
    # Nearest Neighbor
    print("*" * 10 + "1- Nearest Neighbor Implementation" + "*" * 10)
    clf = NN(K)
    clf.fit(Xtr, Ytr)
    Y = clf.predict(Xtt)
    # Evaluate Classification Error
    E = accuracy(Ytt, Y)
    print("Accuracy", E)
    voronoi_plot_2d(Voronoi(Xtr), show_vertices=False, show_points=False, line_colors='orange')
    plt.title(str(clf.K) + "-NN  Implementation Train Data")
    plotit(Xtr, Ytr, clf=clf.predict)
    voronoi_plot_2d(Voronoi(Xtt), show_vertices=False, show_points=False, line_colors='orange')
    plt.title(str(clf.K) + "-NN  Implementation Test data")
    plotit(Xtt, Ytt, clf=clf.predict)
