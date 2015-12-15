import numpy as np
import multiprocessing
import multiprocessing.sharedctypes
import random
from scipy.spatial.distance import cdist
import time
from functools import partial
import pdb


class stochastic_k():
    def __init__(self, n_clusters=8,  max_iter=300, tol=0.00001, n_jobs = 2):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs


    def fit(self, X):

        if self.max_iter <= 0:
            raise ValueError('Number of iterations should be a positive number,'
                             ' got %d instead' % self.max_iter)

        X -= np.mean(X.mean(axis=0))

        init_centers = X[random.sample(np.arange(len(X)), self.n_clusters)]
        # parallelisation of k-means runs
        centers=multiprocessing.Array('d',init_centers.reshape(X.shape[1]*self.n_clusters,1),lock=False)
        processes = []
        for n in xrange(self.n_jobs):
            processes.append(multiprocessing.Process(target=self._kmeans_single, args=(X, centers, self.max_iter, self.n_clusters, self.tol)))
            processes[-1].start()
        for p in processes:
            p.join()
        final_labels, final_inertia = self._labels_inertia(X, centers)
        return centers, final_labels, final_inertia


    def _labels_inertia(self, X, centers):
        """E step of the K-means EM algorithm.
        Compute the labels and the inertia of the given samples and centers.
        This will compute the distances in-place.
        Parameters
        ----------
        X: float64 array-like or CSR sparse matrix, shape (n_samples, n_features)
            The input samples to assign to the labels.
        centers: float64 array, shape (k, n_features)
            The cluster centers.
        Returns
        -------
        labels: int array of shape(n)
            The resulting assignment
        inertia : float
            Sum of distances of samples to their closest cluster center.
        """
        n_samples = X.shape[0]
        temp_centers = np.array(centers[:])
        dist = cdist(X, temp_centers.reshape(len(temp_centers)/X.shape[1], X.shape[1]), 'euclidean')
        labels = dist.argmin(axis=1)
        inertia = np.sum(np.min(dist, axis=1))

        return labels, inertia

    def _label_single(self, x, centers):
        temp_centers = np.array(centers[:])
        dist = cdist(np.array([x]), temp_centers.reshape(len(temp_centers)/len(x), len(x)), 'euclidean')
        label = dist.argmin(axis=1)[0]
        return label

    def recenter(self, labels, x, centers, center_cnt):
        n_features = len(x)
        centers[labels[0]*n_features:(labels[0]+1)*n_features] += x/center_cnt[labels[0]]
        centers[labels[1]*n_features:(labels[1]+1)*n_features] -= x/center_cnt[labels[1]]
        center_cnt[labels[0]] += 1
        center_cnt[labels[1]] -= 1
    def recenter_all(self, labels, X, n_clusters, centers):
        n_features = X.shape[1]
        for c in xrange(n_clusters):
            centers[c*n_features:(c+1)*n_features] = np.mean(X[np.where(labels==c)[0]], axis=0)

    def count_clusters(self, labels, n_clusters):
        center_cnt = np.zeros(shape=n_clusters)
        for c in xrange(n_clusters):
            center_cnt[c] = np.where(labels==c)[0].shape[0]
        return center_cnt

    def _process_cluster(self, centers, X, center_cnt, labels, ind):
        label = self._label_single(X[ind], centers)
        if not (label==labels[ind]):
            self.recenter((label, labels[ind]), X[ind], centers, center_cnt)



    def _kmeans_single(self, X, centers, max_iter, n_clusters, tol):
        start= time.time()
        labels, inertia = self._labels_inertia(X, centers)
        center_cnt = self.count_clusters(labels, n_clusters)
        self.recenter_all(labels, X, n_clusters, centers)
        n_data=X.shape[0]
        index = np.arange(n_data)
        for iter in xrange(max_iter):
            random.shuffle(index)
            start=time.time()
            mapfunc = partial(self._process_cluster, centers, X, center_cnt, labels)
            map(mapfunc, index[:int(n_data*0.05)])
            inertia_prev = inertia
            _, inertia = self._labels_inertia(X, centers)
            if inertia_prev-inertia < tol:
                break

class normal_k():
    def __init__(self, n_clusters=8,  max_iter=300, tol=0.0001):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol


    def fit(self, X):

        if self.max_iter <= 0:
            raise ValueError('Number of iterations should be a positive number,'
                             ' got %d instead' % self.max_iter)

        X -= np.mean(X.mean(axis=0))

        init_centers = X[random.sample(np.arange(len(X)), self.n_clusters)]
        # parallelisation of k-means runs
        centers=multiprocessing.Array('d',init_centers.reshape(X.shape[1]*self.n_clusters,1),lock=False)
        process = multiprocessing.Process(target=self._kmeans_single, args=(X, centers, self.max_iter, self.n_clusters, self.tol))
        process.start()
        process.join()
        final_labels, final_inertia = self._labels_inertia(X, centers)
        return centers, final_labels, final_inertia


    def _labels_inertia(self, X, centers):
        """E step of the K-means EM algorithm.
        Compute the labels and the inertia of the given samples and centers.
        This will compute the distances in-place.
        Parameters
        ----------
        X: float64 array-like or CSR sparse matrix, shape (n_samples, n_features)
            The input samples to assign to the labels.
        centers: float64 array, shape (k, n_features)
            The cluster centers.
        Returns
        -------
        labels: int array of shape(n)
            The resulting assignment
        inertia : float
            Sum of distances of samples to their closest cluster center.
        """
        n_samples = X.shape[0]
        temp_centers = np.array(centers[:])
        dist = cdist(X, temp_centers.reshape(len(temp_centers)/X.shape[1], X.shape[1]), 'euclidean')
        labels = dist.argmin(axis=1)
        inertia = np.sum(np.min(dist, axis=1))

        return labels, inertia

    def _slow_mean(self, data):
        total = 0
        for d in data:
            total += d
        return total/len(data)

    def recenter_all(self, labels, X, n_clusters, centers):
        n_features = X.shape[1]
        for c in xrange(n_clusters):
            centers[c*n_features:(c+1)*n_features] = np.mean(X[np.where(labels==c)[0]], axis=0)

    def _kmeans_single(self, X, centers, max_iter, n_clusters, tol):
        labels, inertia = self._labels_inertia(X, centers)
        for iter in xrange(max_iter):
            self.recenter_all(labels, X, n_clusters, centers)
            inertia_prev = inertia
            labels, inertia = self._labels_inertia(X, centers)
            if abs(inertia-inertia_prev) < tol:
                break
