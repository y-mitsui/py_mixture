#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "bernoulli_mixture.h"

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
	double *log_bernoulli_params1 = malloc(sizeof(double) * n_dimentions * bernoulli_mixture->n_components);
	for(int i=0; i < n_dimentions * bernoulli_mixture->n_components ; i++) {
		log_bernoulli_params1[i] = log(min(max(bernoulli_params[i], 1e-10), 1 - 1e-10));
	}
	double *log_bernoulli_params0 = malloc(sizeof(double) * n_dimentions * bernoulli_mixture->n_components);
	for(int i=0; i < n_dimentions * bernoulli_mixture->n_components ; i++) {
		log_bernoulli_params0[i] = log(min(max(1 - bernoulli_params[i], 1e-10), 1 - 1e-10));
	}
	double *weight_probs = malloc(sizeof(double) * bernoulli_mixture->n_components);
	double loglikelyfood = 0.0;
	for(int i=0; i < n_samples ; i++) {
		for(int j=0; j < bernoulli_mixture->n_components; j++) {
			double bernoulli_prob0 = logBernoulli2(bernoulli_mixture->sample_X0[i],
			                                      bernoulli_mixture->sample_X1[i],
			                                      bernoulli_mixture->n_success[i],
			                                      n_dimentions,
			                                      &log_bernoulli_params0[j * n_dimentions],
			                                      &log_bernoulli_params1[j * n_dimentions]);
            //double bernoulli_prob1 = logBernoulli(&sample_X[i * n_dimentions], n_dimentions, &log_bernoulli_params0[j * n_dimentions], &log_bernoulli_params1[j * n_dimentions]);
			weight_probs[j] = (log_weights[j] + bernoulli_prob0);
		}

		double tot_log_likelyhood = logsumexp(weight_probs, bernoulli_mixture->n_components);
		loglikelyfood += tot_log_likelyhood;
		for(int j=0; j < bernoulli_mixture->n_components; j++) {
			latent_z[i * bernoulli_mixture->n_components + j] = exp(weight_probs[j] - tot_log_likelyhood);
		}
	}
	printf("loglikelyfood:%f\n", loglikelyfood);
	free(log_bernoulli_params0);
	free(log_bernoulli_params1);
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
			for (int i=0; i < bernoulli_mixture->n_success_dim[d]; i++){
				tot_mul += latent_z[bernoulli_mixture->sample_X1_dim[d][i] * bernoulli_mixture->n_components + k];
			}

			bernoulli_params[k * n_dimentions + d] = tot_mul / tot_latent_zk;
		}
		weights[k] = tot_latent_zk / n_samples;
	}
}

void bernoulliMixtureFit(BernoulliMixture *bernoulli_mixture, double *sample_X, int n_samples, int n_dimentions) {
	double *bernoulli_params = malloc(sizeof(double) * bernoulli_mixture->n_components * n_dimentions);
	for (int i=0; i < bernoulli_mixture->n_components * n_dimentions; i++) {
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
    int *tmp0 = malloc(sizeof(int) * n_dimentions);
    int *tmp1 = malloc(sizeof(int) * n_dimentions);
    for (int i=0; i < n_samples; i++) {
        int n_success_row = 0;
        int n_fail_row = 0;
        for (int j=0; j < n_dimentions; j++) {
            if (sample_X[i * n_dimentions + j] == 1.) {
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
    
    bernoulli_mixture->sample_X1_dim = malloc(sizeof(int*) * n_dimentions);
    bernoulli_mixture->n_success_dim = malloc(sizeof(int) * n_dimentions);
    int *tmp1_dim = malloc(sizeof(int) * n_samples);
    for (int j=0; j < n_dimentions; j++) {
        int n_success_row = 0;
        for (int i=0; i < n_samples; i++) {
            if (sample_X[i * n_dimentions + j] == 1.) {
                tmp1_dim[n_success_row++] = i;
            }
        }
        bernoulli_mixture->n_success_dim[j] = n_success_row;
        bernoulli_mixture->sample_X1_dim[j] = malloc(sizeof(int) * n_success_row);
        memcpy(bernoulli_mixture->sample_X1_dim[j], tmp1_dim, sizeof(int) * n_success_row);
    }
    
	for(int iter=0; iter < bernoulli_mixture->n_iter; iter++) {
		bernoulliEStep(bernoulli_mixture, sample_X, n_samples, n_dimentions, bernoulli_params, weights, latent_z);
		bernoulliMStep(bernoulli_mixture, sample_X, n_samples, n_dimentions, bernoulli_params, weights, latent_z);
		if (iter % (bernoulli_mixture->n_iter / 100 + 1) == 0) {
			printf("%d / %d\n", iter, bernoulli_mixture->n_iter);
		}
	}

	bernoulli_mixture->latent_z = latent_z;
	bernoulli_mixture->n_dimentions = n_dimentions;
	bernoulli_mixture->bernoulli_params = bernoulli_params;
	free(weights);
	free(tmp0);
	free(tmp1);
	//free(bernoulli_params);
}
/*
#define N_SAMPLES 10000
#define N_DIMENTIONS 100
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
int main(void) {
    gsl_rng *rng = gsl_rng_alloc (gsl_rng_default);
    for(j=0; j < N_DIMENTIONS; j++) {
        dirichlet_alpha[j] = 1.0;
    }
    gsl_ran_dirichlet(rng, N_DIMENTIONS, dirichlet_alpha, true_prob);
    for(i=0; i < N_SAMPLES; i++) {
        gsl_ran_multinomial(rng, N_DIMENTIONS, 1, true_prob, multi_variate);
        for(j=0; j < N_DIMENTIONS; j++) {
            if(multi_variate[j] == 1)
                sample[j]++;
        }
    }
    
}*/

