
typedef struct {
	int n_components;
	int n_bernoulli_dimentions;
	int n_normal_dimentions;
	double *weights;
	double *bernoulli_params;
	double *normal_means;
	double *normal_sigmas;
	double *sample_bernoulli;
	double *sample_normal;
	double *latent_z;
	int n_iter;
	int **sample_X0;
	int **sample_X1;
	int *n_success;
	int **sample_X1_dim;
	int *n_success_dim;
}BernoulliNormalMixture;

BernoulliNormalMixture *bernoulliNormalMixtureInit(int n_components, int n_iter);
void bernoulliNormalMixtureFit(BernoulliNormalMixture *bernoulli_mixture, double *sample_bernoulli, double *sample_normal, int n_samples, int n_bernoulli_dimentions, int n_normal_dimentions, double *normal_means_init);

