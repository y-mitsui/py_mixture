from libc.stdlib cimport malloc, free

cdef extern from "bernoulli_mixture.h":
    BernoulliMixture *bernoulliMixtureInit(int n_components, int n_iter)
    void bernoulliMixtureFit(BernoulliMixture *bernoulli_mixture, int **success_dimentions, int *n_success, int n_samples, int n_dimentions)
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
    
    def getWeights(self):
        r = []
        for i in range(self.bernoulli_mixture.n_components):
            r.append(self.bernoulli_mixture.weights[i])
        return r
        
    def fit_transform(self, success_dimentions, n_dimentions):
        cdef int n_samples = len(success_dimentions)
        cdef int **c_success_dimentions = <int**>malloc(sizeof(int*) * n_samples)
        cdef int *c_n_cusscess = <int*>malloc(sizeof(int) * n_samples)
        for i in range(n_samples):
            c_n_cusscess[i] = len(success_dimentions[i])
            c_success_dimentions[i] = <int*>malloc(sizeof(int) * c_n_cusscess[i])
            for j in range(c_n_cusscess[i]):
                c_success_dimentions[i][j] = success_dimentions[i][j]
                
        bernoulliMixtureFit(self.bernoulli_mixture, c_success_dimentions, c_n_cusscess, n_samples, n_dimentions)
        result = []
        for i in range(n_samples):
            row = []
            for j in range(self.bernoulli_mixture.n_components):
                row.append(self.bernoulli_mixture.latent_z[i * self.bernoulli_mixture.n_components + j])
            result.append(row)
        free(c_success_dimentions)
        free(c_n_cusscess)
        return result
