from libc.stdlib cimport malloc, free
import numpy as np
from sklearn.cluster import KMeans
import sys

cdef extern from "many_mixture.h":
    ManyMixture *manyMixtureInit(int n_components, int n_iter)
    void manyMixtureFit(ManyMixture *bernoulli_mixture, double *sample_poisson, int **poisson_indexes, int **poisson_counts, int *poisson_n_positive, double *sample_bernoulli, double *sample_normal, int n_samples, int n_poisson_dimentions, int n_bernoulli_dimentions, int n_normal_dimentions, double *normal_means_init)
    ctypedef struct ManyMixture:
        int n_components
        int n_bernoulli_dimentions
        double *latent_z
        double *bernoulli_params
        double *weights
    
cdef class ManyMixtureWrap:
    cdef ManyMixture *bernoulli_mixture
    cdef object init_kmeans
    cdef object n_components
    
    def __init__(self, n_components=2, n_iter=100, init_kmeans=False):
        self.bernoulli_mixture = manyMixtureInit(n_components, n_iter)
        self.init_kmeans = init_kmeans
        self.n_components = n_components
        
    def getMeans(self):
        r = []
        for i in range(self.bernoulli_mixture.n_components):
            row = []
            for j in range(self.bernoulli_mixture.n_bernoulli_dimentions):
                row.append(self.bernoulli_mixture.bernoulli_params[i * self.bernoulli_mixture.n_bernoulli_dimentions + j])
            r.append(row)
        return r
        
    def fit_transform(self, poisson_indexes=None, poisson_counts=None, sample_bernoulli=None, sample_normal=None, normal_mean_init=None):
        cdef int n_samples = 0
        cdef int n_poisson_dimentions = 0
        cdef int i, j, n_col
        
        cdef int **c_poisson_indexes = NULL
        cdef int **c_poisson_counts = NULL
        cdef int *n_positive = NULL
        if poisson_indexes is not None:
            n_samples = len(poisson_indexes)
            c_poisson_indexes = <int**>malloc(sizeof(int*) * n_samples)
            c_poisson_counts = <int**>malloc(sizeof(int*) * n_samples)
            n_positive = <int*>malloc(sizeof(int) * n_samples)
            for i in range(n_samples):
                n_positive[i] = len(poisson_indexes[i])
                c_poisson_indexes[i] = <int*>malloc(sizeof(int) * n_positive[i])
                c_poisson_counts[i] = <int*>malloc(sizeof(int) * n_positive[i])
                for j in range(n_positive[i]):
                    c_poisson_indexes[i][j] = poisson_indexes[i][j]
                    c_poisson_counts[i][j] = poisson_counts[i][j]
                    if n_poisson_dimentions <  c_poisson_indexes[i][j]:
                        n_poisson_dimentions = c_poisson_indexes[i][j]
            n_poisson_dimentions += 1
        
        cdef double *c_sample_bernoulli = NULL
        cdef int n_bernoulli_dimentions = 0
        
        if sample_bernoulli is not None:
            n_samples = sample_bernoulli.shape[0]
            n_bernoulli_dimentions = sample_bernoulli.shape[1]
            c_sample_bernoulli = <double*>malloc(sizeof(double) * n_samples * n_bernoulli_dimentions)
            for i in range(n_samples):
                for j in range(n_bernoulli_dimentions):
                    c_sample_bernoulli[i * n_bernoulli_dimentions + j] = sample_bernoulli[i, j]
                
        cdef int n_normal_dimentions = 0
        cdef double *c_sample_normal = NULL
        cdef double *normal_means_init = NULL
        if sample_normal is not None:
            n_samples = sample_normal.shape[0]
            n_normal_dimentions = sample_normal.shape[1]
            c_sample_normal = <double*>malloc(sizeof(double) * n_samples * n_normal_dimentions)
            for i in range(n_samples):
                for j in range(n_normal_dimentions):
                    c_sample_normal[i * n_normal_dimentions + j] = sample_normal[i, j]
                    
            if self.init_kmeans:
                k_means = KMeans(self.n_components)
                k_means.fit(sample_normal)
                normal_means_init = <double*>malloc(sizeof(double) * k_means.cluster_centers_.shape[0] * k_means.cluster_centers_.shape[1]);
                for k in range(k_means.cluster_centers_.shape[0]):
                    for d in range(k_means.cluster_centers_.shape[1]):
                        normal_means_init[k * k_means.cluster_centers_.shape[1] + d] = k_means.cluster_centers_[k, d]
                
        if n_samples == 0:
            raise Exception("must be number of samples > 0")
                
        manyMixtureFit(self.bernoulli_mixture, NULL, c_poisson_indexes, c_poisson_counts, n_positive, c_sample_bernoulli, c_sample_normal, n_samples, n_poisson_dimentions, n_bernoulli_dimentions, n_normal_dimentions, normal_means_init)
        result = []
        for i in range(n_samples):
            row = []
            for j in range(self.bernoulli_mixture.n_components):
                row.append(self.bernoulli_mixture.latent_z[i * self.bernoulli_mixture.n_components + j])
            result.append(row)
        free(c_sample_bernoulli)
        return result
    
    def fit_predict(self, sample_poisson=None, sample_bernoulli=None, sample_normal=None):
        result = self.fit_transform(sample_poisson, sample_bernoulli, sample_normal)
        return np.argmax(result, 1)
    
        
