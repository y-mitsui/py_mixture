from libc.stdlib cimport malloc, free

cdef extern from "bernoulli_normal_mixture.h":
    BernoulliNormalMixture *bernoulliNormalMixtureInit(int n_components, int n_iter)
    void bernoulliNormalMixtureFit(BernoulliNormalMixture *bernoulli_mixture, double *sample_bernoulli, double *sample_normal, int n_samples, int n_bernoulli_dimentions, int n_normal_dimentions)
    ctypedef struct BernoulliNormalMixture:
        int n_components
        int n_bernoulli_dimentions
        double *latent_z
        double *bernoulli_params
        double *weights
    
cdef class BernoulliNormalMixtureWrap:
    cdef BernoulliNormalMixture *bernoulli_mixture
    
    def __init__(self, n_components, n_iter):
        self.bernoulli_mixture = bernoulliNormalMixtureInit(n_components, n_iter)

    def getMeans(self):
        r = []
        for i in range(self.bernoulli_mixture.n_components):
            row = []
            for j in range(self.bernoulli_mixture.n_bernoulli_dimentions):
                row.append(self.bernoulli_mixture.bernoulli_params[i * self.bernoulli_mixture.n_bernoulli_dimentions + j])
            r.append(row)
        return r
        
    def fit_transform(self, sample_bernoulli, sample_normal):
        cdef int n_samples = sample_bernoulli.shape[0]
        cdef int n_bernoulli_dimentions = sample_bernoulli.shape[1]
        cdef double *c_sample_bernoulli = <double*>malloc(sizeof(double) * n_samples * n_bernoulli_dimentions)
        for i in range(n_samples):
            for j in range(n_bernoulli_dimentions):
                c_sample_bernoulli[i * n_bernoulli_dimentions + j] = sample_bernoulli[i, j]
                
        cdef int n_normal_dimentions = sample_bernoulli.shape[1]
        cdef double *c_sample_normal = <double*>malloc(sizeof(double) * n_samples * n_normal_dimentions)
        for i in range(n_samples):
            for j in range(n_normal_dimentions):
                c_sample_normal[i * n_normal_dimentions + j] = sample_normal[i, j]
                
        bernoulliNormalMixtureFit(self.bernoulli_mixture, c_sample_bernoulli, c_sample_normal, n_samples, n_bernoulli_dimentions, n_normal_dimentions)
        result = []
        for i in range(n_samples):
            row = []
            for j in range(self.bernoulli_mixture.n_components):
                row.append(self.bernoulli_mixture.latent_z[i * self.bernoulli_mixture.n_components + j])
            result.append(row)
        free(c_sample_bernoulli)
        return result
    
