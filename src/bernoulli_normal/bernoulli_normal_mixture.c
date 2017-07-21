#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "bernoulli_normal_mixture.h"

double logsumexp(double *nums, size_t ct) {
  double max_exp = nums[0], sum = 0.0;
  size_t i;

  for (i = 1 ; i < ct ; i++)
    if (nums[i] > max_exp)
      max_exp = nums[i];

  for (i = 0; i < ct ; i++) {
    sum += exp(nums[i] - max_exp);
  }
  return log(sum) + max_exp;
}

double xor128(){
    static unsigned int x=123456789,y=362436069,z=521288629,w=88675123;
    unsigned int t;
    t=(x^(x<<11));
    x=y;
    y=z;
    z=w;
    w=(w^(w>>19))^(t^(t>>8));
    return (double)w/(double)0xFFFFFFFF;
}

BernoulliNormalMixture *bernoulliNormalMixtureInit(int n_components, int n_iter) {
	BernoulliNormalMixture *bernoulli_normal_mixture = malloc(sizeof(BernoulliNormalMixture));
	bernoulli_normal_mixture->n_components = n_components;
	bernoulli_normal_mixture->n_iter = n_iter;
	return bernoulli_normal_mixture;
}

double logBernoulli(double *sample_X, int n_dimentions, double *log_bernoulli_params0, double *log_bernoulli_params1) {
	double log_prob = 0.0;
	
	for(int i=0; i < n_dimentions; i++) {
		log_prob += sample_X[i] * log_bernoulli_params1[i] + (1 - sample_X[i]) * log_bernoulli_params0[i];
	}
	return log_prob;
}

double logBernoulli2(int *sample_X0, int *sample_X1, int n_success, int n_dimentions, double *log_bernoulli_params0, double *log_bernoulli_params1) {
	double log_prob = 0.0;
	for(int i=0; i < n_success; i++) {
	    log_prob += log_bernoulli_params1[sample_X1[i]];
	}
	for(int i=0; i < n_dimentions - n_success; i++) {
	    log_prob += log_bernoulli_params0[sample_X0[i]];
	}
	return log_prob;
}

double logNormal(double *sample_X, int n_dimentions, double *means, double *sigmas) {
	double coef = log(1 / sqrt(2 * M_PI));
	double pdf = 0.0;
	for(int i=0; i < n_dimentions; i++) {
		double coef2 = log(1 / sqrt(sigmas[i]));
		pdf += coef + coef2 - ((sample_X[i] - means[i]) * (sample_X[i] - means[i])) / (2 * sigmas[i]);
	}
	return pdf;
}

double min(double x, double y) {
	return (x < y) ? x : y;
}

double max(double x, double y) {
	return (x > y) ? x : y;
}

void bernoulliNormalEStep(BernoulliNormalMixture *bernoulli_mixture, int n_samples, double *weights, double *latent_z) {
	int n_bernoulli_dimentions = bernoulli_mixture->n_bernoulli_dimentions;
	int n_normal_dimentions = bernoulli_mixture->n_normal_dimentions;
	double *bernoulli_params = bernoulli_mixture->bernoulli_params;

	puts("p0.1");
	double *log_weights = malloc(sizeof(double) * bernoulli_mixture->n_components);
	for(int i=0; i < bernoulli_mixture->n_components ; i++) {
		log_weights[i] = log(max(weights[i], 1e-10));
	}
	puts("p0.2");
	double *log_bernoulli_params1 = malloc(sizeof(double) * n_bernoulli_dimentions * bernoulli_mixture->n_components);
	for(int i=0; i < n_bernoulli_dimentions * bernoulli_mixture->n_components ; i++) {
		log_bernoulli_params1[i] = log(min(max(bernoulli_params[i], 1e-10), 1 - 1e-10));
	}
	double *log_bernoulli_params0 = malloc(sizeof(double) * n_bernoulli_dimentions * bernoulli_mixture->n_components);
	for(int i=0; i < n_bernoulli_dimentions * bernoulli_mixture->n_components ; i++) {
		log_bernoulli_params0[i] = log(min(max(1 - bernoulli_params[i], 1e-10), 1 - 1e-10));
	}
	puts("p0.3");
	double *weight_probs = malloc(sizeof(double) * bernoulli_mixture->n_components);
	double loglikelyfood = 0.0;
	for(int i=0; i < n_samples ; i++) {
		for(int j=0; j < bernoulli_mixture->n_components; j++) {

			double bernoulli_prob0 = logBernoulli2(bernoulli_mixture->sample_X0[i],
			                                      bernoulli_mixture->sample_X1[i],
			                                      bernoulli_mixture->n_success[i],
			                                      n_bernoulli_dimentions,
			                                      &log_bernoulli_params0[j * n_bernoulli_dimentions],
			                                      &log_bernoulli_params1[j * n_bernoulli_dimentions]);

			double normal_prob0 = logNormal(&bernoulli_mixture->sample_normal[i * n_normal_dimentions], n_normal_dimentions, &bernoulli_mixture->normal_means[j * n_normal_dimentions], &bernoulli_mixture->normal_sigmas[j * n_normal_dimentions]);
			weight_probs[j] = (log_weights[j] + bernoulli_prob0 + normal_prob0);
		}
		double tot_log_likelyhood = logsumexp(weight_probs, bernoulli_mixture->n_components);
		loglikelyfood += tot_log_likelyhood;
		for(int j=0; j < bernoulli_mixture->n_components; j++) {
			latent_z[i * bernoulli_mixture->n_components + j] = exp(weight_probs[j] - tot_log_likelyhood);
		}
	}
	puts("p0.6");
	printf("loglikelyfood:%f\n", loglikelyfood);
	free(log_bernoulli_params0);
	free(log_bernoulli_params1);
	free(log_weights);
	free(weight_probs);
}

void bernoulliNormalMStep(BernoulliNormalMixture *bernoulli_mixture, int n_samples, double *weights, double *latent_z) {
	int n_bernoulli_dimentions = bernoulli_mixture->n_bernoulli_dimentions;
	int n_normal_dimentions = bernoulli_mixture->n_normal_dimentions;
	double *bernoulli_params = bernoulli_mixture->bernoulli_params;

	for(int k=0; k < bernoulli_mixture->n_components; k++) {
		puts("p1.0");
		double tot_latent_zk = 0.0;
		for (int i=0; i < n_samples; i++)
			tot_latent_zk += latent_z[i * bernoulli_mixture->n_components + k];

		for(int d=0; d < n_bernoulli_dimentions; d++) {
			double tot_mul = 0.0;
			for (int i=0; i < bernoulli_mixture->n_success_dim[d]; i++){
				tot_mul += latent_z[bernoulli_mixture->sample_X1_dim[d][i] * bernoulli_mixture->n_components + k];
			}
			bernoulli_params[k * n_bernoulli_dimentions + d] = tot_mul / tot_latent_zk;
		}

		puts("p1.1");
		for(int d=0; d < n_normal_dimentions; d++) {
			double tot_mul = 0.0;
			for (int i=0; i < n_samples; i++){
				tot_mul += bernoulli_mixture->sample_normal[i * n_normal_dimentions + d] * latent_z[i * bernoulli_mixture->n_components + k];
			}
			bernoulli_mixture->normal_means[k * n_bernoulli_dimentions + d] = tot_mul / tot_latent_zk;
		}

		puts("p1.2");
		for(int d=0; d < n_normal_dimentions; d++) {
			double tot_mul = 0.0;
			for (int i=0; i < n_samples; i++){
				double diff = bernoulli_mixture->sample_normal[i * n_normal_dimentions + d] - bernoulli_mixture->normal_means[k * n_bernoulli_dimentions + d];
				double mse = diff * diff;
				tot_mul += mse * latent_z[i * bernoulli_mixture->n_components + k];
			}
			bernoulli_mixture->normal_sigmas[k * n_normal_dimentions + d] = max(tot_mul / tot_latent_zk, 1e-8);
		}
		weights[k] = tot_latent_zk / n_samples;
	}
}

void bernoulliNormalMixtureFit(BernoulliNormalMixture *bernoulli_mixture, double *sample_bernoulli, double *sample_normal, int n_samples, int n_bernoulli_dimentions, int n_normal_dimentions, double *normal_means_init) {
	double *bernoulli_params = malloc(sizeof(double) * bernoulli_mixture->n_components * n_bernoulli_dimentions);
	for (int i=0; i < bernoulli_mixture->n_components * n_bernoulli_dimentions; i++) {
		bernoulli_params[i] = (double)rand() / RAND_MAX;
	}
	double *latent_z = malloc(sizeof(double) * n_samples * bernoulli_mixture->n_components);
	double *weights = malloc(sizeof(double) * bernoulli_mixture->n_components);
	double tot_weights = 0.0;
	for (int i=0; i < bernoulli_mixture->n_components; i++) {
		weights[i] = (double)rand() / RAND_MAX;
		tot_weights += weights[i];
	}
	for (int i=0; i < bernoulli_mixture->n_components; i++)
		weights[i] /= tot_weights;

    bernoulli_mixture->sample_X0 = malloc(sizeof(int*) * n_samples);
    bernoulli_mixture->sample_X1 = malloc(sizeof(int*) * n_samples);
    bernoulli_mixture->n_success = malloc(sizeof(int) * n_samples);
    int *tmp0 = malloc(sizeof(int) * n_bernoulli_dimentions);
    int *tmp1 = malloc(sizeof(int) * n_bernoulli_dimentions);
    for (int i=0; i < n_samples; i++) {
        int n_success_row = 0;
        int n_fail_row = 0;
        for (int j=0; j < n_bernoulli_dimentions; j++) {
            if (sample_bernoulli[i * n_bernoulli_dimentions + j] == 1.) {
                tmp1[n_success_row++] = j;
            }else {
                tmp0[n_fail_row++] = j;
            }
        }
        bernoulli_mixture->n_success[i] = n_success_row;
        bernoulli_mixture->sample_X0[i] = malloc(sizeof(int) * n_fail_row);
        bernoulli_mixture->sample_X1[i] = malloc(sizeof(int) * n_success_row);
        memcpy(bernoulli_mixture->sample_X0[i], tmp0, sizeof(int) * n_fail_row);
        memcpy(bernoulli_mixture->sample_X1[i], tmp1, sizeof(int) * n_success_row);
    }

    double *normal_means = malloc(sizeof(double) * bernoulli_mixture->n_components * n_normal_dimentions);
    if(normal_means_init == NULL) {
    	for (int i=0; i < bernoulli_mixture->n_components * n_normal_dimentions; i++) {
			normal_means[i] = (double)rand() / RAND_MAX;
		}
    } else {
    	memcpy(normal_means, normal_means_init, sizeof(double) * bernoulli_mixture->n_components * n_normal_dimentions);
    }

	double *normal_sigmas = malloc(sizeof(double) * bernoulli_mixture->n_components * n_normal_dimentions);
	for (int i=0; i < bernoulli_mixture->n_components * n_normal_dimentions; i++) {
		normal_sigmas[i] = (double)rand() / RAND_MAX * 2 + 10;
	}

    bernoulli_mixture->sample_normal = sample_normal;
    bernoulli_mixture->normal_means = normal_means;
    bernoulli_mixture->normal_sigmas = normal_sigmas;
    bernoulli_mixture->bernoulli_params = bernoulli_params;
    bernoulli_mixture->n_bernoulli_dimentions = n_bernoulli_dimentions;
    bernoulli_mixture->n_normal_dimentions = n_normal_dimentions;
    bernoulli_mixture->sample_X1_dim = malloc(sizeof(int*) * n_bernoulli_dimentions);
    bernoulli_mixture->n_success_dim = malloc(sizeof(int) * n_bernoulli_dimentions);
    int *tmp1_dim = malloc(sizeof(int) * n_samples);
    for (int j=0; j < n_bernoulli_dimentions; j++) {
        int n_success_row = 0;
        for (int i=0; i < n_samples; i++) {
            if (sample_bernoulli[i * n_bernoulli_dimentions + j] == 1.) {
                tmp1_dim[n_success_row++] = i;
            }
        }
        bernoulli_mixture->n_success_dim[j] = n_success_row;
        bernoulli_mixture->sample_X1_dim[j] = malloc(sizeof(int) * n_success_row);
        memcpy(bernoulli_mixture->sample_X1_dim[j], tmp1_dim, sizeof(int) * n_success_row);
    }
	for(int iter=0; iter < bernoulli_mixture->n_iter; iter++) {
		puts("p0");
		bernoulliNormalEStep(bernoulli_mixture, n_samples, weights, latent_z);
		puts("p1");
		bernoulliNormalMStep(bernoulli_mixture, n_samples, weights, latent_z);
		puts("p2");
		if (iter % (bernoulli_mixture->n_iter / 100 + 1) == 0) {
			printf("%d / %d\n", iter, bernoulli_mixture->n_iter);
		}
	}

	bernoulli_mixture->latent_z = latent_z;
	bernoulli_mixture->n_bernoulli_dimentions = n_bernoulli_dimentions;
	bernoulli_mixture->bernoulli_params = bernoulli_params;
	free(weights);
	free(tmp0);
	free(tmp1);
	//free(bernoulli_params);
}

#define N_SAMPLES 1000
#define N_BERNOULLI_DIMENTIONS 100
#define N_NORMAL_DIMENTIONS 10
#define N_COMPONENTS 2
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>

int main(void) {
	int i, j;
	double true_prob[N_COMPONENTS];
	double dirichlet_alpha[N_COMPONENTS];
	unsigned int multi_variate[N_COMPONENTS];
	gsl_rng *rng = gsl_rng_alloc (gsl_rng_default);
	double *bernoulli_means, *normal_means, *normal_sigmas;
	bernoulli_means = malloc(sizeof(double) * N_COMPONENTS * N_BERNOULLI_DIMENTIONS);
	normal_means = malloc(sizeof(double) * N_COMPONENTS * N_NORMAL_DIMENTIONS);
	normal_sigmas = malloc(sizeof(double) * N_COMPONENTS * N_NORMAL_DIMENTIONS);

	for(j=0; j < N_COMPONENTS; j++) {
		dirichlet_alpha[j] = 1.0;
	}
	gsl_ran_dirichlet(rng, N_COMPONENTS, dirichlet_alpha, true_prob);

	for (i=0;i < N_COMPONENTS; i++) {
		for(j=0; j < N_BERNOULLI_DIMENTIONS; j++) {
			bernoulli_means[i * N_BERNOULLI_DIMENTIONS + j] = gsl_rng_uniform(rng);
		}
		for(j=0; j < N_NORMAL_DIMENTIONS; j++) {
			normal_means[i * N_NORMAL_DIMENTIONS + j] = gsl_rng_uniform(rng);
			normal_sigmas[i * N_NORMAL_DIMENTIONS + j] = gsl_rng_uniform(rng) * 2 + 5;
		}
	}

	double *sample_bernoulli, *sample_normal;
	sample_bernoulli = malloc(sizeof(double) * N_SAMPLES * N_BERNOULLI_DIMENTIONS);
	sample_normal = malloc(sizeof(double) * N_SAMPLES * N_NORMAL_DIMENTIONS);

	for(i=0; i < N_SAMPLES; i++) {
		gsl_ran_multinomial(rng, N_COMPONENTS, 1, true_prob, multi_variate);
		int latent_z;
		for(j=0; j < N_COMPONENTS; j++) {
			if(multi_variate[j] == 1) {
				latent_z = j;
				break;
			}
		}
		for(j=0; j < N_BERNOULLI_DIMENTIONS; j++) {
			int success = gsl_ran_bernoulli(rng, bernoulli_means[latent_z * N_BERNOULLI_DIMENTIONS + j]);
			sample_bernoulli[i * N_BERNOULLI_DIMENTIONS + j] = success;
		}
		for(j=0; j < N_NORMAL_DIMENTIONS; j++) {
			double X = gsl_ran_gaussian(rng, normal_sigmas[latent_z * N_NORMAL_DIMENTIONS + j]) + normal_means[latent_z * N_NORMAL_DIMENTIONS + j];
			sample_normal[i * N_NORMAL_DIMENTIONS + j] = X;
		}
	}
	BernoulliNormalMixture *bernoulli_normal_mixture = bernoulliNormalMixtureInit(2, 100);
	bernoulliNormalMixtureFit(bernoulli_normal_mixture, sample_bernoulli, sample_normal, N_SAMPLES, N_BERNOULLI_DIMENTIONS, N_NORMAL_DIMENTIONS, NULL);
}

