
typedef struct {
	int n_components;
	int n_dimentions;
	double *weights;
	double *bernoulli_params;
	double *c_sample_X;
	double *latent_z;
	int n_iter;
	int **sample_X0;
	int **sample_X1;
	int *n_success;
	int **sample_X1_dim;
	int *n_success_dim;
}BernoulliMixture;

BernoulliMixture *bernoulliMixtureInit(int n_components, int n_iter);
void bernoulliMixtureFit(BernoulliMixture *bernoulli_mixture, double *sample_X, int n_samples, int n_dimentions);

