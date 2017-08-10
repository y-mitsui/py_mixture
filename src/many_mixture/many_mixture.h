typedef struct {
	int n_cache;
	double *cache;
}LogFactorial;

typedef struct {
	LogFactorial *log_factorial;
	int n_cache;
	int n_dimentions;
	double *cache;
}LogPoisson;

typedef struct {
	int n_components;
	int n_poisson_dimentions;
	int n_bernoulli_dimentions;
	int n_normal_dimentions;
	double *weights;
	double *poisson_means;
	double *bernoulli_params;
	double *normal_means;
	double *normal_sigmas;
	int **poisson_indexes;
	int **poisson_counts;
	int **poisson_zeros;
	double *sample_poisson;
	double *sample_bernoulli;
	double *sample_normal;
	double *latent_z;
	int n_iter;
	int **sample_X0;
	int **sample_X1;
	int *n_success;
	int **sample_X1_dim;
	int *n_success_dim;
	LogPoisson **log_poisson;
	int **poisson_indexes_dim;
	int **poisson_counters_dim;
	int *poisson_n_positive;
	int *poisson_n_positive_dim;
	
}ManyMixture;

ManyMixture *manyMixtureInit(int n_components, int n_iter);
void manyMixtureFit(ManyMixture *bernoulli_mixture, double *sample_poisson, int **poisson_indexes, int **poisson_counts, int *poisson_n_positive, double *sample_bernoulli, double *sample_normal, int n_samples, int n_poisson_dimentions, int n_bernoulli_dimentions, int n_normal_dimentions, double *normal_means_init);

