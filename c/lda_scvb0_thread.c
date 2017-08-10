#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <err.h>
#include <pthread.h>
#include <unistd.h>
#include "lda.h"

float updateGamma(float *gamma, float *temp_iter, float *nTheta, float alpha, int n_topic);
void weightMean(float *array1, float *array2, int n_sample, float weight1, float weight2);

float updateGammaC(float *gamma, float *temp_iter, float *nTheta, float alpha, int n_topic) {
	double sum_gamma = 0.;
	for(int k=0; k < n_topic; k++){
		gamma[k] = temp_iter[k] *  (nTheta[k] + alpha);
		sum_gamma += gamma[k];
	}
	return sum_gamma;
}

#define d_malloc(type, numb) (type*)dmalloc(sizeof(type) * numb)
#define d_calloc(type, numb) (type*)dcalloc(sizeof(type), numb)

void* dmalloc(size_t size) {
	void *r = malloc(size);
	if (r == NULL) {
		fprintf(stderr, "out of memory\n");
		exit(1);
	}
	return r;
}

void* dcalloc(size_t numb, size_t size) {
	void *r = calloc(numb, size);
	if (r == NULL) {
		fprintf(stderr, "out of memory\n");
		exit(1);
	}
	return r;
}

pthread_mutex_t inference_lock = PTHREAD_MUTEX_INITIALIZER;

int getCpuNum()
{
	return sysconf(_SC_NPROCESSORS_CONF);
}

unsigned long xor128(){ 
    static unsigned long x=123456789,y=362436069,z=521288629,w=88675123; 
    unsigned long t; 
    t=(x^(x<<11));x=y;y=z;z=w; return( w=(w^(w>>19))^(t^(t>>8)) ); 
} 

void scvb0Save(Scvb0 *ctx, const char *path){
    FILE *fp;
    
    if (!(fp=fopen(path,"wb"))) {
        perror(path);
        return;
    }
    fwrite(&ctx->n_topic, sizeof(int), 1, fp);
    fwrite(&ctx->n_iter, sizeof(int), 1, fp);
    fwrite(&ctx->n_document, sizeof(int), 1, fp);
    fwrite(&ctx->batch_size, sizeof(int), 1, fp);
    fwrite(&ctx->n_word_type, sizeof(int), 1, fp);
    fwrite(&ctx->n_all_word, sizeof(long long), 1, fp);
    fwrite(&ctx->alpha, sizeof(float), 1, fp);
    fwrite(&ctx->beta, sizeof(float), 1, fp);
    
    //fwrite(ctx->nTheta, sizeof(float), ctx->n_document * ctx->n_topic, fp);
    fwrite(ctx->Phi, sizeof(float), ctx->n_word_type * ctx->n_topic, fp);
    //fwrite(ctx->nz, sizeof(float), ctx->n_topic, fp);
    fclose(fp);
    
}

Scvb0 *scvb0Load(const char *path) {
    FILE *fp;
    Scvb0 *ctx = d_malloc(Scvb0, 1);
    
    if(!(fp = fopen(path/*"data/scvb_data.dat"*/,"rb"))) {
        return NULL;
    }
    fread(&ctx->n_topic, sizeof(int), 1, fp);
    fread(&ctx->n_iter, sizeof(int), 1, fp);
    fread(&ctx->n_document, sizeof(int), 1, fp);
    fread(&ctx->batch_size, sizeof(int), 1, fp);
    fread(&ctx->n_word_type, sizeof(int), 1, fp);
    fread(&ctx->n_all_word, sizeof(long long), 1, fp);
    fread(&ctx->alpha, sizeof(float), 1, fp);
    fread(&ctx->beta, sizeof(float), 1, fp);
    
    //ctx->gamma = d_malloc(float, ctx->n_topic);
    //ctx->nzHat = d_malloc(float, ctx->n_topic);
    //ctx->nPhiHat = d_malloc(float, ctx->n_word_type * ctx->n_topic);
    //ctx->nz = d_malloc(float, ctx->n_topic);
    //ctx->nPhi = d_malloc(float, ctx->n_word_type * ctx->n_topic);
    //ctx->nTheta = malloc(sizeof(float) * ctx->n_document * ctx->n_topic);
    ctx->Phi = d_malloc(float, ctx->n_word_type * ctx->n_topic);
    //ctx->Theta = malloc(sizeof(float) * ctx->n_document * ctx->n_topic);
    
    //fwrite(ctx->nTheta, sizeof(float), ctx->n_document * ctx->n_topic, fp);
    fread(ctx->Phi, sizeof(float), ctx->n_word_type * ctx->n_topic, fp);
    //fread(ctx->nz, sizeof(float), ctx->n_topic, fp);
    fclose(fp);
    return ctx;
}

void scvb0InitTest(int n_topic, int n_iter, int batch_size, float alpha, float beta){
    Scvb0 *res = d_malloc(Scvb0, 1);
    
    res->n_topic = n_topic;
    res->n_iter = n_iter;
    res->batch_size = batch_size;
    res->alpha = alpha;
    res->beta = beta;
    
}

Scvb0* scvb0Init(int n_topic, int n_iter, int batch_size, int n_thread, float alpha, float beta) {
    Scvb0 *res = d_malloc(Scvb0, 1);

    res->n_topic = n_topic;
    res->n_iter = n_iter;
    res->batch_size = batch_size;
    res->n_thread = (n_thread < 0) ? getCpuNum() : n_thread;
    res->alpha = alpha;
    res->beta = beta;

    return res;
}

void scvb0Free(Scvb0* ctx) {
    /*free(ctx->gamma);
    free(ctx->nzHat);
    free(ctx->nPhiHat);
    free(ctx->nz);
    free(ctx->nPhi);*/
    free(ctx->Phi);
    free(ctx);
}

void scvb0EstPhi(Scvb0 *ctx, float *Phi) {
    int k, v;
    for (k = 0; k < ctx->n_topic; k++) {
        float normSum = 0;
        for (v = 0; v < ctx->n_word_type; v++) {
            normSum += ctx->nPhi[v * ctx->n_topic + k] + ctx->beta;
        }
        for (v = 0; v < ctx->n_word_type; v++) {
            Phi[v * ctx->n_topic + k] = (ctx->nPhi[v * ctx->n_topic + k] + ctx->beta) / normSum;
        }
    }
}

void scvb0EstTheta(Scvb0 *ctx, float *Theta) {
    int d;
    int k;
    float k_sum;
    
    for(d=0; d < ctx->n_document; d++){
        k_sum = 0.;
        for(k=0;k<ctx->n_topic;k++){
            k_sum += ctx->nTheta[d * ctx->n_topic + k] + ctx->alpha;
        }
        k_sum = 1. / k_sum ;
    
        for(k=0;k<ctx->n_topic;k++){
            Theta[d * ctx->n_topic + k] = (ctx->alpha + ctx->nTheta[d * ctx->n_topic + k] ) * k_sum;
        }
    }   
}

float perplexity(Scvb0 *ctx, int **word_indexes_ptr, short** word_counts_ptr, int *n_word_type_each_doc){
	float *Theta = d_malloc(float, ctx->n_document * ctx->n_topic);
	float *nPhiHat = d_calloc(float, ctx->n_word_type * ctx->n_topic);
    scvb0EstPhi(ctx, nPhiHat);
    scvb0EstTheta(ctx, Theta);
    
    float log_per = 0.0;
    int N = 0;
    int d, v, k;
    
    for(d=0; d < ctx->n_document; d++){
        for(v=0; v < n_word_type_each_doc[d]; v++){
            int term = word_indexes_ptr[d][v];
            short freq = word_counts_ptr[d][v];
            float k_sum = 0.;
            for(k=0;k<ctx->n_topic;k++){
                k_sum += nPhiHat[term * ctx->n_topic + k] * Theta[d * ctx->n_topic + k];
            }
            
            log_per -= log(k_sum) * freq;
            N += freq;
        }
    }
    free(Theta);
    free(nPhiHat);
    return exp(log_per / N);
}

float *scvb0TransformSingle(Scvb0 *ctx, int *doc_word, int n_word, int max_iter){
    int i, j, k;
        
    float *pzs = d_calloc(float, n_word * ctx->n_topic);
    float *pzs_new = d_malloc(float, n_word * ctx->n_topic);
    for (i=0; i < max_iter; i++) {
        for (j=0; j < n_word; j++) {
            float pzs_sum = 0;
            for (k=0; k < ctx->n_topic; k++) {
                pzs_sum += pzs[j * ctx->n_topic + k];
            }
            for (k=0; k < ctx->n_topic; k++) {
                pzs_new[j * ctx->n_topic + k] = ctx->Phi[doc_word[j] * ctx->n_topic + k] * (pzs_sum - pzs[j * ctx->n_topic + k] + ctx->alpha);
            }
        }
        for (j=0; j < n_word; j++) {
            float pzs_new_sum = 0;
            for (k=0; k < ctx->n_topic; k++) {
                pzs_new_sum += pzs_new[j * ctx->n_topic + k];
            }
            for (k=0; k < ctx->n_topic; k++) {
                pzs_new[j * ctx->n_topic + k] /= pzs_new_sum;
            }
        }
        float delta_naive = 0;
        for (j=0; j < n_word; j++) {
            for (k=0; k < ctx->n_topic; k++) {
                delta_naive += pzs_new[j * ctx->n_topic + k] - pzs[j * ctx->n_topic + k];
            }
        }
        
        memcpy(pzs, pzs_new, sizeof(float) * n_word * ctx->n_topic);
    }
    float *result = d_calloc(float, ctx->n_topic);
    for (k=0; k < ctx->n_topic; k++) {
        for (j=0; j < n_word; j++) {
            result[k] += pzs_new[j * ctx->n_topic + k];
        }
    }
    free(pzs);
    free(pzs_new);
    return result;
}

static void scvb0Infer(Scvb0 *ctx, int** word_indexes_ptr, short** word_counts_ptr, int* n_word_each_doc, int* n_word_type_each_doc,int doc_id_offset, int n_document, int *index_offset){
    int i, j, d, v, k;
    float sum_gamma;
    float batch_size_coef = 1. / n_document;
    float *temp_iter = d_malloc(float, ctx->n_topic);
    float *temp_iterA = d_malloc(float, ctx->n_topic);
    float *nPhiHat = d_calloc(float, ctx->n_word_type * ctx->n_topic);
    float *nzHat = d_calloc(float, ctx->n_topic);
    float *gamma = d_malloc(float, ctx->n_topic);
    
    for(k=0; k < ctx->n_topic; k++){
        temp_iterA[k] = 1. / (ctx->nz[k] + ctx->beta * ctx->n_word_type);
    }

    for(d=0; d < n_document; d++){
        int doc_id = index_offset[d];
        float update_theta_coef = ctx->rhoTheta * n_word_each_doc[doc_id];
        
        for(i=0; i < 1; i++){
            for(v=0; v < n_word_type_each_doc[doc_id]; v++){
                int term = word_indexes_ptr[doc_id][v];
                for(k=0; k < ctx->n_topic; k++){
                    temp_iter[k] = (ctx->nPhi[term * ctx->n_topic + k] + ctx->beta) * temp_iterA[k];
                }
                
                for(j=0; j < word_counts_ptr[doc_id][v]; j++){
                	sum_gamma = updateGammaC(gamma, temp_iter, &ctx->nTheta[doc_id * ctx->n_topic], ctx->alpha, ctx->n_topic);

                    for(k=0; k < ctx->n_topic; k++){
                        gamma[k] /= sum_gamma;
                        ctx->nTheta[doc_id * ctx->n_topic + k] = (1. - ctx->rhoTheta) * ctx->nTheta[doc_id * ctx->n_topic + k] + update_theta_coef * gamma[k];
                    }
                }
            }
        }
        for(v=0; v < n_word_type_each_doc[doc_id]; v++){
            int term = word_indexes_ptr[doc_id][v];
            for(k=0; k < ctx->n_topic; k++){
                temp_iter[k] = (ctx->nPhi[term * ctx->n_topic + k] + ctx->beta) * temp_iterA[k];
            }
            for(i=0; i < word_counts_ptr[doc_id][v]; i++){
            	sum_gamma = updateGammaC(gamma, temp_iter, &ctx->nTheta[doc_id * ctx->n_topic], ctx->alpha, ctx->n_topic);
                for(k=0; k < ctx->n_topic; k++){
                    gamma[k] /= sum_gamma;
                    ctx->nTheta[doc_id * ctx->n_topic + k] = (1. - ctx->rhoTheta) * ctx->nTheta[doc_id * ctx->n_topic + k] + update_theta_coef * gamma[k];
                }
            
                for(k=0; k < ctx->n_topic; k++){
                    nPhiHat[term * ctx->n_topic + k] += ctx->n_all_word * gamma[k] * batch_size_coef;
                    nzHat[k] += ctx->n_all_word * gamma[k] * batch_size_coef;
                }
            }
        }
    }

    /*struct timeval startTime, endTime;
	gettimeofday(&startTime, NULL);
	double a = 1. - ctx->rhoPhi;
	double b = ctx->rhoPhi;
	weightMean(ctx->nPhi, nPhiHat, ctx->n_word_type * ctx->n_topic, a, b);*/
	/*for(i=0; i < ctx->n_word_type * ctx->n_topic; i++)
		ctx->nPhi[i] = a * ctx->nPhi[i] + b * nPhiHat[i];*/

    for(v=0; v < ctx->n_word_type; v++){
        for(k=0; k < ctx->n_topic; k++){
            //if (ctx->nPhiHat[v * ctx->n_topic + k]!=0){
                ctx->nPhi[v * ctx->n_topic + k] = (1. - ctx->rhoPhi) * ctx->nPhi[v * ctx->n_topic + k] + ctx->rhoPhi * nPhiHat[v * ctx->n_topic + k];
            //}
        }
    }
    /*gettimeofday(&endTime, NULL);
	time_t diffsec = difftime(endTime.tv_sec, startTime.tv_sec);
	suseconds_t diffsub = endTime.tv_usec - startTime.tv_usec;
	float realsec = diffsec + diffsub * 1e-6;
	fprintf(stderr, "(%.3lfsec)\n", realsec);*/

    for(k=0; k < ctx->n_topic; k++){
        ctx->nz[k] = (1. - ctx->rhoPhi) * ctx->nz[k] + ctx->rhoPhi * nzHat[k];
    }
    
    free(temp_iter);
    free(temp_iterA);
    free(gamma);
    free(nPhiHat);
    free(nzHat);
}

typedef struct{
    float *nzHat;
    float *nPhiHat;
    float *nThetaHat;
}ThreadScvb0;

void *thread_main(void* arg) {
	ThreadArgs *thread_args = (ThreadArgs*)arg;
	Scvb0 *ctx = thread_args->ctx;
	int *doc_indxes = d_malloc(int, ctx->batch_size);
	
	for(int i=0; i < ctx->n_iter; i++){
		if (i % ctx->n_thread != thread_args->thread_no) continue;
		
		struct timeval startTime, endTime;
		gettimeofday(&startTime, NULL);
		if ((i + 1) % (ctx->n_iter / 5) == 0){
			fprintf(stderr, "perplexity:%f\n", perplexity(ctx, thread_args->word_indexes_ptr, thread_args->word_counts_ptr, thread_args->n_word_type_each_doc));
		}

		ctx->rhoPhi = 10. / pow(1000. + i, 0.9);
		ctx->rhoTheta = 1. / pow(10. + i, 0.9);

		for (int j=0; j < ctx->batch_size; j++){
			doc_indxes[j] = xor128() % thread_args->n_document;
		}
		scvb0Infer(ctx,
					thread_args->word_indexes_ptr,
					thread_args->word_counts_ptr,
					thread_args->n_word_each_doc,
					thread_args->n_word_type_each_doc,
					0,
					ctx->batch_size,
					doc_indxes);
		gettimeofday(&endTime, NULL);
		time_t diffsec = difftime(endTime.tv_sec, startTime.tv_sec);
		suseconds_t diffsub = endTime.tv_usec - startTime.tv_usec;
		float realsec = diffsec + diffsub * 1e-6;
		fprintf(stderr, "%d: %d / %d (%.3lfsec)\n", thread_args->thread_no, i, ctx->n_iter, realsec);
	}
	return 0;
}

void scvb0Fit(Scvb0 *ctx, int** word_indexes_ptr, short** word_counts_ptr, int* n_word_each_doc, int* n_word_type_each_doc, long long n_all_word, int n_document, int n_word_type, float *result){
    int i, j, k, v, d;
    ctx->n_all_word = n_all_word;
    ctx->n_word_type = n_word_type;
    ctx->n_document = n_document;

    ctx->nz = d_malloc(float, ctx->n_topic);
    ctx->nPhi = d_malloc(float, ctx->n_word_type * ctx->n_topic);
    ctx->nTheta = d_malloc(float, n_document * ctx->n_topic);

    for(k=0; k < ctx->n_topic; k++){
        float sum_nPhi = 0.;
        for(v=0; v < ctx->n_word_type; v++){
            ctx->nPhi[v * ctx->n_topic + k] = (float)rand() / RAND_MAX;
            sum_nPhi += ctx->nPhi[v * ctx->n_topic + k];
        }
        ctx->nz[k] = sum_nPhi;
    }
    
    for(d=0; d < n_document; d++){
        for(k=0; k < ctx->n_topic; k++){
            ctx->nTheta[d * ctx->n_topic + k] = (float)rand() / RAND_MAX;
        }
    }

    int *doc_indxes = d_malloc(int, ctx->batch_size);
    
    pthread_t *thread = malloc(sizeof(pthread_t) * ctx->n_thread);
    int  *iret = malloc(sizeof(int) * ctx->n_thread);
    ThreadArgs **thread_args = malloc(sizeof(ThreadArgs *) * ctx->n_thread);
    for (int i=0; i < ctx->n_thread; i++) {
		thread_args[i] = d_malloc(ThreadArgs, 1);
		thread_args[i]->thread_no = i;
		thread_args[i]->word_indexes_ptr = word_indexes_ptr;
		thread_args[i]->word_counts_ptr = word_counts_ptr;
		thread_args[i]->ctx = ctx;
		thread_args[i]->n_word_each_doc = n_word_each_doc;
		thread_args[i]->n_word_type_each_doc = n_word_type_each_doc;
		thread_args[i]->n_document = n_document;
    }
    
    for (int i=0; i < ctx->n_thread; i++) {
    	iret[i] = pthread_create( &thread[i], NULL, thread_main, (void*) thread_args[i]);
    	sleep(rand() % 7 + 1);
    }
    
    for (int i=0; i < ctx->n_thread; i++) {
    	int ret1 = pthread_join(thread[i], NULL);
    	if (ret1 != 0) {
			errc(EXIT_FAILURE, ret1, "can not join thread 1");
		}
    }
    
    for (int i=0; i < ctx->n_thread; i++) {
    	free(thread_args[i]);
    }
    free(thread_args);
    free(iret);
    free(thread);
	
    for(d=0; d < ctx->n_document; d++) {
    	float k_sum =0.0;
    	for(k=0; k < ctx->n_topic; k++) {
    		k_sum += ctx->nTheta[d * ctx->n_topic + k];
    	}
    	for(k=0; k < ctx->n_topic; k++) {
    		result[d * ctx->n_topic + k] = ctx->nTheta[d * ctx->n_topic + k] / k_sum;
    	}
    }

    free(doc_indxes);
    /*free(ctx->gamma);
    free(ctx->nzHat);
    free(ctx->nPhiHat);*/
    free(ctx->nz);
    free(ctx->nTheta);
    ctx->Phi = d_malloc(float, ctx->n_word_type * ctx->n_topic);
    for (k = 0; k < ctx->n_topic; k++) {
        float normSum = 0;
        for (v = 0; v < ctx->n_word_type; v++) {
            normSum += ctx->nPhi[v * ctx->n_topic + k] + ctx->beta;
        }
        for (v = 0; v < ctx->n_word_type; v++) {
            ctx->Phi[v * ctx->n_topic + k] = (ctx->nPhi[v * ctx->n_topic + k] + ctx->beta) / normSum;
        }
    }
    free(ctx->nPhi);
    free(ctx->Phi);
}

void getWordInfo(char *data, int *word_index, short *word_count) {
	char *word_id = data;
	char *n_word = strchr(data, ':');
	*n_word='\0';
	n_word++;
	*word_index = atoi(word_id);
	*word_count = atoi(n_word);
}
/*
int main(void) {
	char buf[500000];
	int n_document = 0;
	FILE *fp = fopen("/Users/mitsuiyosuke/Documents/workspace/recommend_advance/data/VV/lda.csv", "r");
	while(fgets(buf, sizeof(buf), fp)) {
		n_document++;
	}
	fclose(fp);
	int **word_indexes = d_malloc(int*, n_document);
	short **word_counts = d_malloc(short*, n_document);
	int *n_word_type_each_doc = d_malloc(int, n_document);
	int *n_word_each_doc = d_malloc(int, n_document);
	int n_all_word = 0;
	int n_all_word_type = 0;
	int n_topics = 200;

	fp = fopen("/Users/mitsuiyosuke/Documents/workspace/recommend_advance/data/VV/lda.csv", "r");
	int line=0;
	char *p, *p2;
	char **document_ids = malloc(sizeof(char*) * n_document);
	while(fgets(buf, sizeof(buf), fp)) {
		buf[strlen(buf) - 1] = '\0';
		if (buf[0]=='\0') continue;

		int n_word_type = 0;
		for(p=buf;*p && (p2=strchr(p, ','));p=p2 + 1) {
			n_word_type++;
		}
		n_word_type++;

		word_indexes[line] = d_malloc(int, n_word_type);
		word_counts[line] = d_malloc(short, n_word_type);

		p = strchr(buf, ',');
		*p='\0';
		p++;
		document_ids[line] = strdup(buf);

		int n_word_doc = 0;
		n_word_type = 0;
		for(;*p && (p2=strchr(p, ','));p=p2 + 1) {
			*p2 = '\0';
			getWordInfo(p, &word_indexes[line][n_word_type], &word_counts[line][n_word_type]);
			n_word_doc += word_counts[line][n_word_type];
			n_word_type++;
		}
		getWordInfo(p, &word_indexes[line][n_word_type], &word_counts[line][n_word_type]);
		n_word_doc += word_counts[line][n_word_type];
		n_word_type++;

		n_word_type_each_doc[line] = n_word_type;
		n_word_each_doc[line] = n_word_doc;
		n_all_word += n_word_each_doc[line];
		n_all_word_type += n_word_type_each_doc[line];

		line++;
		if (line==2000) break;
	}
	fclose(fp);

	int bath_size = (n_document < 1000) ? n_document : 1000;
	Scvb0* sctx = scvb0Init(n_topics, 400, bath_size, 1, 0.5, 0.5);
	float *topics = d_malloc(float, n_document * n_topics);
	printf("n_document:%d\n", n_document);
	scvb0Fit(sctx, word_indexes, word_counts, n_word_each_doc, n_word_type_each_doc, n_all_word, n_document, n_all_word_type, topics);
	fp=fopen("/Users/mitsuiyosuke/Documents/workspace/recommend_advance/data/VV/lda_result.csv", "w");
	for(int i=0; i < n_document; i++) {
		char tmp[10000];
		sprintf(tmp, ",%s", document_ids[i]);
		for(int j=0; j < n_topics; j++) {
			sprintf(tmp, "%s,%f", tmp, topics[i * n_topics + j]);
		}
		fprintf(fp, "%s\n", &tmp[1]);
	}
	fclose(fp);

	return 0;
}*/

