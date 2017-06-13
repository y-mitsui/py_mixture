#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bernoulli_mixture.h"
float flogsumexp(const float* __restrict__ buf, int N);

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

BernoulliMixture *bernoulliMixtureInit(int n_components, int n_iter) {
	BernoulliMixture *bernoulli_mixture = malloc(sizeof(BernoulliMixture));
	bernoulli_mixture->n_components = n_components;
	bernoulli_mixture->n_iter = n_iter;
	return bernoulli_mixture;
}

double logBernoulli(double *sample_X, int n_dimentions, double *log_bernoulli_params) {
	double log_prob = 0.0;
	for(int i=0; i < n_dimentions; i++) {
		log_prob += sample_X[i] * log_bernoulli_params[i] + (1 - sample_X[i]) * (1 - log_bernoulli_params[i]);
	}
	return log_prob;
}

inline double min(double x, double y) {
	return (x < y) ? x : y;
}

inline double max(double x, double y) {
	return (x > y) ? x : y;
}

void bernoulliEStep(BernoulliMixture *bernoulli_mixture, double *sample_X, int n_samples, int n_dimentions, double *bernoulli_params, double *weights, double *latent_z) {
	double *log_weights = malloc(sizeof(double) * bernoulli_mixture->n_components);
	for(int i=0; i < bernoulli_mixture->n_components ; i++) {
		log_weights[i] = log(max(weights[i], 1e-10));
	}
	double *log_bernoulli_params = malloc(sizeof(double) * n_dimentions * bernoulli_mixture->n_components);
	for(int i=0; i < n_dimentions * bernoulli_mixture->n_components ; i++) {
		log_bernoulli_params[i] = log(min(max(bernoulli_params[i], 1e-10), 1 - 1e-10));
	}
	float *weight_probs = malloc(sizeof(double) * bernoulli_mixture->n_components);
	for(int i=0; i < n_samples ; i++) {
		for(int j=0; j < bernoulli_mixture->n_components; j++) {
			double bernoulli_prob = logBernoulli(&sample_X[i * n_dimentions], n_dimentions, &log_bernoulli_params[j * n_dimentions]);
			weight_probs[j] = (log_weights[j] + bernoulli_prob);
		}

		double tot_log_likelyhood = (double)logsumexp(weight_probs, bernoulli_mixture->n_components);
		for(int j=0; j < bernoulli_mixture->n_components; j++) {
			latent_z[i * bernoulli_mixture->n_components + j] = exp(weight_probs[j] - tot_log_likelyhood);
		}
	}
	free(log_bernoulli_params);
	free(log_weights);
	free(weight_probs);
}

void bernoulliMStep(BernoulliMixture *bernoulli_mixture, double *sample_X, int n_samples, int n_dimentions, double *bernoulli_params, double *weights, double *latent_z) {
	for(int k=0; k < bernoulli_mixture->n_components; k++) {
		double tot_latent_zk = 0.0;
		for (int i=0; i < n_samples; i++)
			tot_latent_zk += latent_z[i * bernoulli_mixture->n_components + k];

		for(int d=0; d < n_dimentions; d++) {
			double tot_mul = 0.0;
			for (int i=0; i < n_samples; i++)
				tot_mul += latent_z[i * bernoulli_mixture->n_components + k] * sample_X[i * n_dimentions + d];

			bernoulli_params[k * n_dimentions + d] = tot_mul / tot_latent_zk;
			weights[k] = tot_latent_zk / n_samples;
		}
	}
}

void bernoulliMixtureFit(BernoulliMixture *bernoulli_mixture, double *sample_X, int n_samples, int n_dimentions) {
	double *bernoulli_params = malloc(sizeof(double) * bernoulli_mixture->n_components * n_dimentions);
	for (int i=0; i < bernoulli_mixture->n_components * n_dimentions; i++) {
		bernoulli_params[i] = xor128();
	}
	double *latent_z = malloc(sizeof(double) * n_samples * bernoulli_mixture->n_components);
	double *weights = malloc(sizeof(double) * bernoulli_mixture->n_components);
	double tot_weights = 0.0;
	for (int i=0; i < bernoulli_mixture->n_components; i++) {
		weights[i] = xor128();
		tot_weights += weights[i];
	}
	for (int i=0; i < bernoulli_mixture->n_components; i++)
		weights[i] /= tot_weights;

	for(int iter=0; iter < bernoulli_mixture->n_iter; iter++) {
		bernoulliEStep(bernoulli_mixture, sample_X, n_samples, n_dimentions, bernoulli_params, weights, latent_z);
		bernoulliMStep(bernoulli_mixture, sample_X, n_samples, n_dimentions, bernoulli_params, weights, latent_z);
		if (iter % (bernoulli_mixture->n_iter / 100 + 1) == 0) {
			printf("%d / %d\n", iter, bernoulli_mixture->n_iter);
		}
	}

	bernoulli_mixture->latent_z = latent_z;
	free(weights);
	free(bernoulli_params);
}


