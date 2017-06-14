#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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

double logBernoulli(int *sample_X, int n_dimentions, double *log_bernoulli_params) {
	double log_prob = 0.0;
	for(int i=0; i < n_dimentions; i++) {
		log_prob += sample_X[i] * log_bernoulli_params[i] + (1 - sample_X[i]) * (1 - log_bernoulli_params[i]);
	}
	return log_prob;
}

#include <xmmintrin.h>
#include <emmintrin.h>
double logBernoulli2(int *sample_X, int n_dimentions, double *log_bernoulli_params) {
	double k_sum[2] = {0.,0.};
	double ones[2] = {1.,1.};
	double tmp[2] = {1.,1.};
	double tmp2[2] = {1.,1.};
	double tmp3[2] = {1.,1.};
	double log_prob = 0.0;

	__m128d w_k_sum = _mm_load_pd(k_sum);
	__m128d w_ones = _mm_load_pd(ones);
	__m128d w_tmp = _mm_load_pd(tmp);
	__m128d w_tmp2 = _mm_load_pd(tmp2);
	__m128d w_tmp3 = _mm_load_pd(tmp3);
	for(int i=0; i < n_dimentions / 2; i++) {
		__m128d w_sample_X = _mm_load_pd(&sample_X[i]);
		__m128d w_log_bernoulli_params = _mm_load_pd(&log_bernoulli_params[i]);
		w_tmp = _mm_mul_pd(w_sample_X, w_log_bernoulli_params);

		w_tmp2 = _mm_sub_pd(w_ones, w_sample_X);
		w_tmp3 = _mm_sub_pd(w_ones, w_log_bernoulli_params);

		w_tmp2 = _mm_mul_pd(w_tmp2, w_tmp3);

		w_tmp = _mm_add_pd(w_tmp, w_tmp2);
		w_k_sum = _mm_add_pd(w_tmp, w_k_sum);
	}
	return k_sum[0] + k_sum[1];
}

int main(void) {
	int n_samples = 10000;
	int n_dimentions = 1000;
	int *sample = malloc(sizeof(int) * n_samples * n_dimentions);
	double *log_bernoulli_params = malloc(sizeof(double) * n_dimentions);
	for (int i=0; i < n_dimentions; i++) {
		log_bernoulli_params[i] = log(xor128());
	}

	for (int i=0; i < n_samples; i++) {
		for (int j=0; j < n_dimentions; j++) {
			sample[i * n_dimentions + j] = rand() % 2;
		}
	}
	time_t t1 = time(NULL);
	puts("start");
	double dammy = 0.0;
	for (int i=0; i < n_samples; i++) {
		for(int i=0;i < 100; i++) {
			dammy += logBernoulli2(&sample[i * n_dimentions], n_dimentions, log_bernoulli_params);
		}
	}
	printf("%dsec\n%f\n", time(NULL) - t1, dammy);
}

