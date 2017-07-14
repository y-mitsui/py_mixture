from libc.stdlib cimport malloc, free

cdef extern from "bernoulli_mixture.h":
    BernoulliMixture *bernoulliMixtureInit(int n_components, int n_iter)
    void bernoulliMixtureFit(BernoulliMixture *bernoulli_mixture, double *sample_X, int n_samples, int n_dimentions)
    ctypedef struct BernoulliMixture:
        int n_components
        int n_dimentions
        double *latent_z
        double *bernoulli_params
        double *weights
    
cdef class BernoulliMixtureWrap:
    cdef BernoulliMixture *bernoulli_mixture
    
    def __init__(self, n_components, n_iter):
        self.bernoulli_mixture = bernoulliMixtureInit(n_components, n_iter)

    def getMeans(self):
        r = []
        for i in range(self.bernoulli_mixture.n_components):
            row = []
            for j in range(self.bernoulli_mixture.n_dimentions):
                row.append(self.bernoulli_mixture.bernoulli_params[i * self.bernoulli_mixture.n_dimentions + j])
            r.append(row)
        return r
        
    def fit_transform(self, sample_X):
        cdef int n_samples = sample_X.shape[0]
        cdef int n_dimentions = sample_X.shape[1]
        cdef double *c_sample_X = <double*>malloc(sizeof(double) * n_samples * n_dimentions)
        for i in range(n_samples):
            for j in range(n_dimentions):
                c_sample_X[i * n_dimentions + j] = sample_X[i, j]
                
        bernoulliMixtureFit(self.bernoulli_mixture, c_sample_X, n_samples, n_dimentions)
        result = []
        for i in range(n_samples):
            row = []
            for j in range(self.bernoulli_mixture.n_components):
                row.append(self.bernoulli_mixture.latent_z[i * self.bernoulli_mixture.n_components + j])
            result.append(row)
        free(c_sample_X)
        return result
    
    def fit_predict(self, sample_X):
        cdef int n_samples = sample_X.shape[0]
        cdef int n_dimentions = sample_X.shape[1]
        cdef double *c_sample_X = <double*>malloc(sizeof(double) * n_samples * n_dimentions)
        for i in range(n_samples):
            for j in range(n_dimentions):
                c_sample_X[i * n_dimentions + j] = sample_X[i, j]
                
        bernoulliMixtureFit(self.bernoulli_mixture, c_sample_X, n_samples, n_dimentions)
        result = []
        for i in range(n_samples):
            row = []
            max_prob = self.bernoulli_mixture.latent_z[i * self.bernoulli_mixture.n_components]
            max_components = 0
            for j in range(1, self.bernoulli_mixture.n_components):
                prob = self.bernoulli_mixture.latent_z[i * self.bernoulli_mixture.n_components + j]
                if max_prob < prob:
                    max_prob = prob
                    max_components = j
            result.append(max_components)
        free(c_sample_X)
        return result
