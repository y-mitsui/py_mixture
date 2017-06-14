
typedef struct {
	int n_components;
	int n_dimentions;
	double *weights;
	double *means;
	double *c_sample_X;
	double *latent_z;
	int n_iter;
	int **sample_X0;
	int **sample_X1;
	int *n_success;
	int **sample_dim1;
	int *n_success_dim;
}BernoulliMixture;

BernoulliMixture *bernoulliMixtureInit(int n_components, int n_iter);
void bernoulliMixtureFit(BernoulliMixture *bernoulli_mixture, int **success_dimentions, int *n_success, int n_samples, int n_dimentions);

