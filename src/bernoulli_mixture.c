#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include "bernoulli_mixture.h"
float flogsumexp(const float* __restrict__ buf, int N);

double logsumexp(double *nums, size_t ct) {
  double max_exp = nums[0], sum = 0.0;
  size_t i;

  for (i = 1 ; i < ct ; i++){
    if (nums[i] > max_exp) {
      max_exp = nums[i];

    }
  }

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

double logBernoulli(int *sample_X0, int *sample_X1, int n_success, int n_dimentions, double *log_bernoulli_params0, double *log_bernoulli_params1) {
	double log_prob = 0.0;

	for(int i=0; i < n_success; i++) {
		log_prob += log_bernoulli_params1[sample_X1[i]] ;
	}
	for(int i=0; i < n_dimentions - n_success; i++) {
		log_prob += log_bernoulli_params0[sample_X0[i]];
	}

	return log_prob;
}

double dmin(double x, double y) {
	return (x < y) ? x : y;
}

double dmax(double x, double y) {
	return (x > y) ? x : y;
}

void bernoulliEStep(BernoulliMixture *bernoulli_mixture, int n_samples, int n_dimentions, double *bernoulli_params, double *weights, double *latent_z) {
	double *log_weights = malloc(sizeof(double) * bernoulli_mixture->n_components);
	for(int i=0; i < bernoulli_mixture->n_components ; i++) {
		log_weights[i] = log(dmax(weights[i], 1e-7));
	}
	double *log_bernoulli_params0 = malloc(sizeof(double) * n_dimentions * bernoulli_mixture->n_components);
	double *log_bernoulli_params1 = malloc(sizeof(double) * n_dimentions * bernoulli_mixture->n_components);
	for(int i=0; i < bernoulli_mixture->n_components ; i++) {
		for(int j=0; j < n_dimentions; j++) {
			log_bernoulli_params0[i * n_dimentions + j] = log(dmin(dmax(bernoulli_params[i * n_dimentions + j], 1e-7), 1 - 1e-7));
			log_bernoulli_params1[i * n_dimentions + j] = log(dmin(dmax(1 - bernoulli_params[i * n_dimentions + j], 1e-7), 1 - 1e-7));
		}
	}
	double *weight_probs = malloc(sizeof(double) * bernoulli_mixture->n_components);
	double loglikelyfood = 0.0;
	for(int i=0; i < n_samples ; i++) {
		for(int j=0; j < bernoulli_mixture->n_components; j++) {
			double bernoulli_prob = logBernoulli(bernoulli_mixture->sample_X0[i], bernoulli_mixture->sample_X1[i], bernoulli_mixture->n_success[i], n_dimentions, &log_bernoulli_params0[j * n_dimentions], &log_bernoulli_params1[j * n_dimentions]);
			weight_probs[j] = (log_weights[j] + bernoulli_prob);
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

void bernoulliMStep(BernoulliMixture *bernoulli_mixture, int n_samples, int n_dimentions, double *bernoulli_params, double *weights, double *latent_z) {
	for(int k=0; k < bernoulli_mixture->n_components; k++) {
		double tot_latent_zk = 0.0;
		for (int i=0; i < n_samples; i++) {
			tot_latent_zk += latent_z[i * bernoulli_mixture->n_components + k];
		}
		for(int d=0; d < n_dimentions; d++) {
			double tot_mul = 0.0;
			for (int i=0; i < bernoulli_mixture->n_success_dim[d]; i++) {
				int smaple_idx = bernoulli_mixture->sample_dim1[d][i];
				tot_mul += latent_z[smaple_idx * bernoulli_mixture->n_components + k];
			}
			bernoulli_params[k * n_dimentions + d] = tot_mul / tot_latent_zk;

			weights[k] = tot_latent_zk / n_samples;
		}
	}
}

void bernoulliMixtureFit(BernoulliMixture *bernoulli_mixture, int **success_dimentions, int *n_success, int n_samples, int n_dimentions) {
	int **sample_X0 = malloc(sizeof(int*) * n_samples);
	int **sample_X1 = malloc(sizeof(int*) * n_samples);

	for (int i=0; i < n_samples; i++) {
		sample_X0[i] = malloc(sizeof(int) * (n_dimentions - n_success[i]));
		sample_X1[i] = malloc(sizeof(int) * n_success[i]);
		int n_0 = 0;
		int n_1 = 0;
		for (int j=0; j < n_dimentions; j++) {
			int k;
			for(k=0; k < n_success[i]; k++) {
				if (success_dimentions[i][k] == j) break;
			}
			if (k < n_success[i]) {
				sample_X1[i][n_1++] = j;
			}else {
				sample_X0[i][n_0++] = j;
			}
		}
	}

	int **sample_dim1 = malloc(sizeof(int*) * n_dimentions);
	int *tmp = malloc(sizeof(int) * n_samples);
	int *n_success_dim = calloc(1, sizeof(int) * n_dimentions);
	for (int j=0; j < n_dimentions; j++) {
		for (int i=0; i < n_samples; i++) {
			int k;
			for(k=0; k < n_success[i]; k++) {
				if (success_dimentions[i][k] == j) break;
			}
			if (k < n_success[i]) {
				tmp[n_success_dim[j]++] = i;
			}
		}
		sample_dim1[j] = malloc(sizeof(int) * n_success_dim[j]);
		memcpy(sample_dim1[j], tmp, sizeof(int) * n_success_dim[j]);
	}
	bernoulli_mixture->sample_dim1 = sample_dim1;
	bernoulli_mixture->n_success_dim = n_success_dim;

	bernoulli_mixture->sample_X0 = sample_X0;
	bernoulli_mixture->sample_X1 = sample_X1;
	bernoulli_mixture->n_success = n_success;
	//srand(time(NULL));
	srand(1234);
	double *bernoulli_params = malloc(sizeof(double) * bernoulli_mixture->n_components * n_dimentions);
	for (int i=0; i < bernoulli_mixture->n_components * n_dimentions; i++) {
		bernoulli_params[i] = (double)rand()/RAND_MAX;
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
		bernoulliEStep(bernoulli_mixture, n_samples, n_dimentions, bernoulli_params, weights, latent_z);
		bernoulliMStep(bernoulli_mixture, n_samples, n_dimentions, bernoulli_params, weights, latent_z);
		//if (iter % (bernoulli_mixture->n_iter / 100 + 1) == 0) {
		/*if (iter % 1 == 0) {
			printf("%d / %d\n", iter, bernoulli_mixture->n_iter);
		}*/
		for(int i=0; i < bernoulli_mixture->n_components; i++) {
			for(int j=0; j < n_dimentions; j++) {
				printf("%.3f ", bernoulli_params[i * n_dimentions + j]);
			}
			puts("");
		}
	}

	bernoulli_mixture->latent_z = latent_z;
	bernoulli_mixture->weights = weights;
	bernoulli_mixture->bernoulli_params = bernoulli_params;
	bernoulli_mixture->n_dimentions = n_dimentions;

	/*for(int i=0; i < bernoulli_mixture->n_components; i++) {
		for(int j=0; j < bernoulli_mixture->n_dimentions; j++) {
			printf("%.3f ", bernoulli_mixture->bernoulli_params[i * bernoulli_mixture->n_dimentions + j]);
		}
		puts("");
	}*/
}


