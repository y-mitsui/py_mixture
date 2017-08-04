#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "many_mixture.h"

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

ManyMixture *manyMixtureInit(int n_components, int n_iter) {
	ManyMixture *bernoulli_normal_mixture = malloc(sizeof(ManyMixture));
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

//階乗
//TODO:スターリングの近似
int factorial(int n) {
    if (n > 0) {
        return n * factorial(n - 1);
    } else {
        return 1;
    }
}

double _logFactorialCompute(int n) {
    if (n > 0) {
        return log(n) + _logFactorialCompute(n - 1);
    } else {
        return log(1);
    }
}

double logFactorialCompute(LogFactorial *log_factorial, int x) {
	if (x < log_factorial->n_cache) return log_factorial->cache[x];
	else return _logFactorialCompute(x);
}

LogFactorial *logFactorialInit(int n_cache) {
	LogFactorial *log_factorial = malloc(sizeof(LogFactorial));
	log_factorial->n_cache = n_cache;
	log_factorial->cache = malloc(sizeof(double) * n_cache);
	for(int i=0; i < log_factorial->n_cache; i++) {
		log_factorial->cache[i] = _logFactorialCompute(i);
	}
	return log_factorial;
}

LogPoisson *logPoissonInit(int n_pdf_cache, int n_dimentions, LogFactorial *log_factorial) {
	LogPoisson *log_poisson = malloc(sizeof(LogPoisson));
	log_poisson->n_cache = n_pdf_cache;
	log_poisson->n_dimentions = n_dimentions;
	log_poisson->log_factorial = log_factorial;
	log_poisson->cache = malloc(sizeof(double) * n_pdf_cache * n_dimentions);
	return log_poisson;
}

void logPoissonCache(LogPoisson *log_poisson, double *means) {
	for(int i=0; i < log_poisson->n_cache; i++) {
		for(int j=0; j < log_poisson->n_dimentions; j++) {
			log_poisson->cache[i * log_poisson->n_dimentions + j] = (i * log(means[j]) - means[j]) - logFactorialCompute(log_poisson->log_factorial, i);
		}
	}
}

double logPoissonPdf(LogPoisson *log_poisson, double *sample_X, int n_dimentions, double *means) {
	double pdf = 0.0;
	for(int i=0; i < n_dimentions; i++) {
		int x = (int)sample_X[i];
		if (x < log_poisson->n_cache)
			pdf += log_poisson->cache[x * log_poisson->n_dimentions + i];
		else
			pdf += (sample_X[i] * log(means[i]) - means[i]) - logFactorialCompute(log_poisson->log_factorial, sample_X[i]);
	}
	return pdf;
}

double logPoissonPdf2(LogPoisson *log_poisson, int *sample_indexes, int *sample_count, int *zero_dimentions, int n_positive, int n_dimentions, double *means) {
	double pdf = 0.0;
	for(int i=0; i < n_positive; i++) {
		int dimention = sample_indexes[i];
		int x = sample_count[i];
		if (x < log_poisson->n_cache)
			pdf += log_poisson->cache[x * log_poisson->n_dimentions + dimention];
		else
			pdf += (x * log(means[dimention]) - means[dimention]) - logFactorialCompute(log_poisson->log_factorial, x);
	}

	for(int i=0; i < n_dimentions - n_positive; i++) {
		int dimention = zero_dimentions[i];
		pdf += log_poisson->cache[dimention];
	}
	return pdf;
}

double logPoisson(double *sample_X, int n_dimentions, double *means) {
	double pdf = 0.0;
	for(int i=0; i < n_dimentions; i++) {
		pdf += (sample_X[i] * log(means[i]) - means[i]) - _logFactorialCompute(sample_X[i]);
	}
	return pdf;
}

double min(double x, double y) {
	return (x < y) ? x : y;
}

double max(double x, double y) {
	return (x > y) ? x : y;
}

void bernoulliNormalEStep(ManyMixture *bernoulli_mixture, int n_samples, double *weights, double *latent_z) {
	int n_poisson_dimentions = bernoulli_mixture->n_poisson_dimentions;
	int n_bernoulli_dimentions = bernoulli_mixture->n_bernoulli_dimentions;
	int n_normal_dimentions = bernoulli_mixture->n_normal_dimentions;
	double *bernoulli_params = bernoulli_mixture->bernoulli_params;

	for(int i=0; i < bernoulli_mixture->n_components ; i++) {
		logPoissonCache(bernoulli_mixture->log_poisson[i], &bernoulli_mixture->poisson_means[i * n_poisson_dimentions]);
	}

	double *log_weights = malloc(sizeof(double) * bernoulli_mixture->n_components);
	for(int i=0; i < bernoulli_mixture->n_components ; i++) {
		log_weights[i] = log(max(weights[i], 1e-10));
	}
	double *log_bernoulli_params1 = malloc(sizeof(double) * n_bernoulli_dimentions * bernoulli_mixture->n_components);
	for(int i=0; i < n_bernoulli_dimentions * bernoulli_mixture->n_components ; i++) {
		log_bernoulli_params1[i] = log(min(max(bernoulli_params[i], 1e-10), 1 - 1e-10));
	}
	double *log_bernoulli_params0 = malloc(sizeof(double) * n_bernoulli_dimentions * bernoulli_mixture->n_components);
	for(int i=0; i < n_bernoulli_dimentions * bernoulli_mixture->n_components ; i++) {
		log_bernoulli_params0[i] = log(min(max(1 - bernoulli_params[i], 1e-10), 1 - 1e-10));
	}
	double *weight_probs = malloc(sizeof(double) * bernoulli_mixture->n_components);
	double loglikelyfood = 0.0;
	for(int i=0; i < n_samples ; i++) {
		for(int j=0; j < bernoulli_mixture->n_components; j++) {
			//double poisson_prob = logPoisson(&bernoulli_mixture->sample_poisson[i * n_poisson_dimentions], n_poisson_dimentions, &bernoulli_mixture->poisson_means[j * n_poisson_dimentions]);
			//double poisson_prob0 = logPoissonPdf(bernoulli_mixture->log_poisson[j], &bernoulli_mixture->sample_poisson[i * n_poisson_dimentions], n_poisson_dimentions, &bernoulli_mixture->poisson_means[j * n_poisson_dimentions]);
			double poisson_prob = logPoissonPdf2(bernoulli_mixture->log_poisson[j], bernoulli_mixture->poisson_indexes[i], bernoulli_mixture->poisson_counts[i], bernoulli_mixture->poisson_zeros[i], bernoulli_mixture->poisson_n_positive[i], n_poisson_dimentions, &bernoulli_mixture->poisson_means[j * n_poisson_dimentions]);
			double bernoulli_prob = logBernoulli2(bernoulli_mixture->sample_X0[i],
			                                      bernoulli_mixture->sample_X1[i],
			                                      bernoulli_mixture->n_success[i],
			                                      n_bernoulli_dimentions,
			                                      &log_bernoulli_params0[j * n_bernoulli_dimentions],
			                                      &log_bernoulli_params1[j * n_bernoulli_dimentions]);
			double normal_prob = logNormal(&bernoulli_mixture->sample_normal[i * n_normal_dimentions], n_normal_dimentions, &bernoulli_mixture->normal_means[j * n_normal_dimentions], &bernoulli_mixture->normal_sigmas[j * n_normal_dimentions]);
			weight_probs[j] = (log_weights[j] + bernoulli_prob + normal_prob + poisson_prob);
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

void bernoulliNormalMStep(ManyMixture *bernoulli_mixture, int n_samples, double *weights, double *latent_z) {
	int n_poisson_dimentions = bernoulli_mixture->n_poisson_dimentions;
	int n_bernoulli_dimentions = bernoulli_mixture->n_bernoulli_dimentions;
	int n_normal_dimentions = bernoulli_mixture->n_normal_dimentions;
	double *poisson_means = bernoulli_mixture->poisson_means;
	double *bernoulli_params = bernoulli_mixture->bernoulli_params;

	for(int k=0; k < bernoulli_mixture->n_components; k++) {
		double tot_latent_zk = 0.0;
		for (int i=0; i < n_samples; i++)
			tot_latent_zk += latent_z[i * bernoulli_mixture->n_components + k];

		/*for(int d=0; d < n_poisson_dimentions; d++) {
			double tot_mul = 0.0;
			for (int i=0; i < n_samples; i++){
				tot_mul += bernoulli_mixture->sample_poisson[i * n_poisson_dimentions + d] * latent_z[i * bernoulli_mixture->n_components + k];
			}
			poisson_means[k * n_poisson_dimentions + d] = tot_mul / tot_latent_zk;
		}*/
		for(int d=0; d < n_poisson_dimentions; d++) {
		    double tot_mul = 0.0;
			for (int i=0; i < bernoulli_mixture->poisson_n_positive_dim[d]; i++){
			    int val = bernoulli_mixture->poisson_indexes_dim[d][i];
				tot_mul += bernoulli_mixture->poisson_counters_dim[d][i] * latent_z[val * bernoulli_mixture->n_components + k];
			}
			
			/*double tot_mul2 = 0.0;
			for (int i=0; i < n_samples; i++){
				tot_mul2 += bernoulli_mixture->sample_poisson[i * n_poisson_dimentions + d] * latent_z[i * bernoulli_mixture->n_components + k];
			}*/
			poisson_means[k * n_poisson_dimentions + d] = tot_mul / tot_latent_zk;
		}
		/*for(int d=0; d < n_poisson_dimentions; d++) {
			double tot_mul = 0.0;
			for (int i=0; i < bernoulli_mixture->n_word_type_dim[d]; i++){
			    int val = bernoulli_mixture->poisson_index_dim[d][i];
				tot_mul += bernoulli_mixture->poisson_conter_dim[d][i] *  val * latent_z[val * bernoulli_mixture->n_components + k];
			}
			poisson_means[k * n_poisson_dimentions + d] = tot_mul / tot_latent_zk;
		}*/


		for(int d=0; d < n_bernoulli_dimentions; d++) {
			double tot_mul = 0.0;
			for (int i=0; i < bernoulli_mixture->n_success_dim[d]; i++){
				tot_mul += latent_z[bernoulli_mixture->sample_X1_dim[d][i] * bernoulli_mixture->n_components + k];
			}
			bernoulli_params[k * n_bernoulli_dimentions + d] = tot_mul / tot_latent_zk;
		}

		for(int d=0; d < n_normal_dimentions; d++) {
			double tot_mul = 0.0;
			for (int i=0; i < n_samples; i++){
				tot_mul += bernoulli_mixture->sample_normal[i * n_normal_dimentions + d] * latent_z[i * bernoulli_mixture->n_components + k];
			}
			bernoulli_mixture->normal_means[k * n_normal_dimentions + d] = tot_mul / tot_latent_zk;
		}

		for(int d=0; d < n_normal_dimentions; d++) {
			double tot_mul = 0.0;
			for (int i=0; i < n_samples; i++){
				double diff = bernoulli_mixture->sample_normal[i * n_normal_dimentions + d] - bernoulli_mixture->normal_means[k * n_normal_dimentions + d];
				double mse = diff * diff;
				tot_mul += mse * latent_z[i * bernoulli_mixture->n_components + k];
			}
			bernoulli_mixture->normal_sigmas[k * n_normal_dimentions + d] = max(tot_mul / tot_latent_zk, 1e-8);
		}
		weights[k] = tot_latent_zk / n_samples;
	}
}

typedef struct _List{
	struct _List *next;
	int index;
	int count;
}List;

List *listAlloc() {
	List *list = malloc(sizeof(List));
	list->index = -1;
	list->next = NULL;
	return list;
}

List *listAdd(List *list, int index, int count) {
	list->index = index;
	list->count = count;
	list->next = listAlloc();
	return list->next;
}

void manyMixtureFit(ManyMixture *bernoulli_mixture, double *sample_poisson, int **poisson_indexes, int **poisson_counts, int *poisson_n_positive, double *sample_bernoulli, double *sample_normal, int n_samples, int n_poisson_dimentions, int n_bernoulli_dimentions, int n_normal_dimentions, double *normal_means_init) {
	double *poisson_means = malloc(sizeof(double) * bernoulli_mixture->n_components * n_poisson_dimentions);
	for (int i=0; i < bernoulli_mixture->n_components * n_poisson_dimentions; i++) {
		poisson_means[i] = (double)rand() / RAND_MAX * 2;
	}
	double *bernoulli_params = malloc(sizeof(double) * bernoulli_mixture->n_components * n_bernoulli_dimentions);
	for (int i=0; i < bernoulli_mixture->n_components * n_bernoulli_dimentions; i++) {
		bernoulli_params[i] = (double)rand() / RAND_MAX;
	}

	LogFactorial *log_factorial = logFactorialInit(100);
	bernoulli_mixture->log_poisson = malloc(sizeof(LogPoisson) * bernoulli_mixture->n_components);
	for(int i=0; i < bernoulli_mixture->n_components; i++)
		bernoulli_mixture->log_poisson[i] = logPoissonInit(10, n_poisson_dimentions, log_factorial);

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

	bernoulli_mixture->sample_poisson = sample_poisson;
	bernoulli_mixture->poisson_indexes = poisson_indexes;
	bernoulli_mixture->poisson_counts = poisson_counts;
	bernoulli_mixture->poisson_n_positive = poisson_n_positive;
    bernoulli_mixture->sample_normal = sample_normal;
    bernoulli_mixture->normal_means = normal_means;
    bernoulli_mixture->normal_sigmas = normal_sigmas;
    bernoulli_mixture->bernoulli_params = bernoulli_params;
    bernoulli_mixture->poisson_means = poisson_means;
    bernoulli_mixture->n_poisson_dimentions = n_poisson_dimentions;
    bernoulli_mixture->n_bernoulli_dimentions = n_bernoulli_dimentions;
    bernoulli_mixture->n_normal_dimentions = n_normal_dimentions;

    time_t t0 = time(NULL);
    bernoulli_mixture->poisson_zeros = malloc(sizeof(int*) * n_samples);
    int *sample_row = malloc(sizeof(int) * n_poisson_dimentions);
    //int **poisson_indexes_dim = malloc(sizeof(int*) * n_poisson_dimentions);
    List **poission_indexes_dim_first = malloc(sizeof(List*) * n_poisson_dimentions);
    List **poission_indexes_dim_cur = malloc(sizeof(List*) * n_poisson_dimentions);
    for(int j=0; j < n_poisson_dimentions; j++) {
    	poission_indexes_dim_cur[j] = poission_indexes_dim_first[j] = listAlloc();
    }
    int *n_positive = calloc(1, sizeof(int) * n_poisson_dimentions);
	for(int i=0; i < n_samples; i++) {
		bernoulli_mixture->poisson_zeros[i] = malloc(sizeof(int) * (n_poisson_dimentions - poisson_n_positive[i]));
		int n_zeros = 0;
		memset(sample_row, 0, sizeof(int) * n_poisson_dimentions);
		for(int j=0; j < poisson_n_positive[i]; j++) {
			sample_row[poisson_indexes[i][j]] = poisson_counts[i][j];
		}
		for(int j=0; j < n_poisson_dimentions; j++) {
			if (sample_row[j] != 0) {
				poission_indexes_dim_cur[j] = listAdd(poission_indexes_dim_cur[j], i, sample_row[j]);
				n_positive[j]++;
			}else {
			    bernoulli_mixture->poisson_zeros[i][n_zeros++] = j;
			}
		}
	}
	free(sample_row);
	printf("%d\n", time(NULL) - t0);
	int **poisson_indexes_dim = malloc(sizeof(int*) * n_poisson_dimentions);
	int **poisson_counters_dim = malloc(sizeof(int*) * n_poisson_dimentions);
	int *positive_sample_index = malloc(sizeof(int) * n_samples);
	int *positive_sample_counter = malloc(sizeof(int) * n_samples);
    for (int j=0; j < n_poisson_dimentions; j++) {
    	poisson_indexes_dim[j] = malloc(sizeof(int) * n_positive[j]);
		poisson_counters_dim[j] = malloc(sizeof(int) * n_positive[j]);
		int n=0;
    	for (List* cur = poission_indexes_dim_first[j]; cur && cur->index >= 0; cur = cur->next, n++) {
    		poisson_indexes_dim[j][n] = cur->index;
    		poisson_counters_dim[j][n] = cur->count;
    	}

		//memcpy(poisson_indexes_dim[j], positive_sample_index, sizeof(int) * n_positive[j]);
		//memcpy(poisson_counters_dim[j], positive_sample_counter, sizeof(int) * n_positive[j]);
    }
    printf("%d\n", time(NULL) - t0);
    bernoulli_mixture->poisson_indexes_dim = poisson_indexes_dim;
    bernoulli_mixture->poisson_counters_dim = poisson_counters_dim;
    bernoulli_mixture->poisson_n_positive_dim = n_positive;
    /*int max_value = 0;
    for (int j=0; j < n_poisson_dimentions; j++) {
		for (int i=0; i < n_samples; i++) {
			if (max_value < (int)sample_poisson[i * n_poisson_dimentions + j]) {
				max_value = (int)sample_poisson[i * n_poisson_dimentions + j];
			}
		}
    }
    int *n_word_type_dim = calloc(1, sizeof(int) * n_poisson_dimentions);
    int *poisson_index = malloc(sizeof(int) * (max_value + 1));
    for (int j=0; j < n_poisson_dimentions; j++) {
    	int n_types = 0;
    	memset(poisson_index, -1, sizeof(int) * (max_value + 1));
		for (int i=0; i < n_samples; i++) {
			int val = (int)sample_poisson[i * n_poisson_dimentions + j];
			if (poisson_index[val] == -1) poisson_index[val] = n_types++;
		}
		n_word_type_dim[j] = n_types;
    }
    int **poisson_index_dim = malloc(sizeof(int*) * n_poisson_dimentions);
    int **poisson_conter_dim = malloc(sizeof(int*) * n_poisson_dimentions);
    for (int j=0; j < n_poisson_dimentions; j++) {
    	int n_types = 0;
    	memset(poisson_index, -1, sizeof(int) * max_value);
    	poisson_index_dim[j] = malloc(sizeof(int) * n_word_type_dim[j]);
		poisson_conter_dim[j] = calloc(1, sizeof(int) * n_word_type_dim[j]);

    	for (int i=0; i < n_samples; i++) {
    		int val = (int)sample_poisson[i * n_poisson_dimentions + j];
    		if (poisson_index[val] == -1) poisson_index[val] = n_types++;
    		poisson_index_dim[j][poisson_index[val]] = val;
    		poisson_conter_dim[j][poisson_index[val]]++;
    	}
    }
    bernoulli_mixture->poisson_index_dim = poisson_index_dim;
    bernoulli_mixture->poisson_conter_dim = poisson_conter_dim;
    bernoulli_mixture->n_word_type_dim = n_word_type_dim;*/

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
    time_t t1 = time(NULL);
	for(int iter=0; iter < bernoulli_mixture->n_iter; iter++) {
		bernoulliNormalEStep(bernoulli_mixture, n_samples, weights, latent_z);
		bernoulliNormalMStep(bernoulli_mixture, n_samples, weights, latent_z);
		//if (iter % (bernoulli_mixture->n_iter / 100 + 1) == 0) {
		    printf("%d / %d (%d)\n", iter, bernoulli_mixture->n_iter, time(NULL) - t1);
		    t1 = time(NULL);
		//}
	}

	bernoulli_mixture->latent_z = latent_z;
	bernoulli_mixture->n_bernoulli_dimentions = n_bernoulli_dimentions;
	bernoulli_mixture->bernoulli_params = bernoulli_params;
	free(weights);
	free(tmp0);
	free(tmp1);
	//free(bernoulli_params);
}

#define N_SAMPLES 35000
#define N_BERNOULLI_DIMENTIONS 1
#define N_POISSON_DIMENTIONS 35000
#define N_NORMAL_DIMENTIONS 1
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
	double *poisson_means, *bernoulli_means, *normal_means, *normal_sigmas;
	poisson_means = malloc(sizeof(double) * N_COMPONENTS * N_POISSON_DIMENTIONS);
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
		for(j=0; j < N_POISSON_DIMENTIONS; j++) {
			poisson_means[i * N_POISSON_DIMENTIONS + j] = gsl_rng_uniform(rng) * 0.1;
		}
		for(j=0; j < N_NORMAL_DIMENTIONS; j++) {
			normal_means[i * N_NORMAL_DIMENTIONS + j] = gsl_rng_uniform(rng);
			normal_sigmas[i * N_NORMAL_DIMENTIONS + j] = gsl_rng_uniform(rng) * 2 + 5;
		}
	}

	double *sample_poisson, *sample_bernoulli, *sample_normal;
	//sample_poisson = malloc(sizeof(double) * N_SAMPLES * N_POISSON_DIMENTIONS);
	sample_bernoulli = malloc(sizeof(double) * N_SAMPLES * N_BERNOULLI_DIMENTIONS);
	sample_normal = malloc(sizeof(double) * N_SAMPLES * N_NORMAL_DIMENTIONS);
	int **poisson_indexes, **poisson_counts;
	int *n_positives;
	poisson_indexes = malloc(sizeof(int*) * N_SAMPLES);
	poisson_counts = malloc(sizeof(int*) * N_SAMPLES);
	n_positives = malloc(sizeof(int) * N_SAMPLES);
	int *indexes_row = malloc(sizeof(int) * N_POISSON_DIMENTIONS);
	int *counts_row = malloc(sizeof(int) * N_POISSON_DIMENTIONS);
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
		int n_positive_row = 0;
		for(j=0; j < N_POISSON_DIMENTIONS; j++) {
			int x = gsl_ran_poisson(rng, poisson_means[latent_z * N_POISSON_DIMENTIONS + j]);
			//sample_poisson[i * N_POISSON_DIMENTIONS + j] = (double)x;
			if (x > 0) {
    			indexes_row[n_positive_row] = j;
	    		counts_row[n_positive_row] = x;
	    		n_positive_row++;
    		}
		}
		n_positives[i] = n_positive_row;
		poisson_indexes[i] = malloc(sizeof(int) * n_positive_row);
		poisson_counts[i] = malloc(sizeof(int) * n_positive_row);
		memcpy(poisson_indexes[i], indexes_row, sizeof(int) * n_positive_row);
		memcpy(poisson_counts[i], counts_row, sizeof(int) * n_positive_row);


		for(j=0; j < N_NORMAL_DIMENTIONS; j++) {
			double X = gsl_ran_gaussian(rng, normal_sigmas[latent_z * N_NORMAL_DIMENTIONS + j]) + normal_means[latent_z * N_NORMAL_DIMENTIONS + j];
			sample_normal[i * N_NORMAL_DIMENTIONS + j] = X;
		}
	}
	ManyMixture *bernoulli_normal_mixture = manyMixtureInit(2, 100);
	manyMixtureFit(bernoulli_normal_mixture, sample_poisson, poisson_indexes, poisson_counts, n_positives, sample_bernoulli, sample_normal, N_SAMPLES, N_POISSON_DIMENTIONS, N_BERNOULLI_DIMENTIONS, N_NORMAL_DIMENTIONS, NULL);

}

