#ifndef __LDA_CVB0_H__
#define __LDA_CVB0_H__

#ifdef __cplusplus
extern "C" {
#endif
typedef struct{
    float ***gamma_dik;
    int n_topics;
    int n_iter;
    int n_thread;
    float* alpha;
    float alpha_sum;
    float beta;
    int** word_indexes_ptr;
    float* nd;
    short** word_counts_ptr;
    int* n_word_each_doc;
    int n_document;
    int n_words;
    float* mean_ndk;
    float* mean_nvk;
    float* mean_nk;
    int optimize_alpha;
    int slirent;
}Ldacvb0;

Ldacvb0* Ldacvb0Init(int n_topics,int n_iter,float *alpha ,float beta);
void Ldacvb0InitTest(int n_topics,int n_iter,float *alpha ,float beta);
float *Ldacvb0FitTransform(Ldacvb0* ctx, int **word_indexes, short **word_counts, int n_document, int n_word_type, int *n_word_each_doc);
float *Ldacvb0EstPhi(Ldacvb0* ctx);



typedef struct {
    int n_topic;
    int n_iter;    
    int n_document;
    int batch_size;
    int n_word_type;
    int n_thread;
    long long n_all_word;
    
    float alpha;
    float beta;
    
    float rhoPhi;
    float rhoTheta;
    
    float *nTheta;
    float *nPhi;
    float *nz;
    float *Phi;
    float *Theta;
    
    float *gamma;
    float *nzHat;
    float *nPhiHat;
    
}Scvb0;

typedef struct {
	int thread_no;
	int *doc_indxes;
	Scvb0 *ctx;
	int n_document;
	int** word_indexes_ptr;
	short** word_counts_ptr;
	int* n_word_each_doc;
	int* n_word_type_each_doc;
}ThreadArgs;

Scvb0* scvb0Init(int n_topic, int n_iter, int batch_size, int n_thread, float alpha, float beta);
void scvb0Fit(Scvb0 *ctx, int** word_indexes_ptr, short** word_counts_ptr, int* n_word_each_doc, int* n_word_type_each_doc,long long n_all_word, int n_document, int n_word_type, float *result);
void scvb0Save(Scvb0 *ctx, const char *path);
Scvb0 *scvb0Load(const char *path);
void scvb0EstPhi(Scvb0 *ctx, float *Phi);
float *scvb0TransformSingle(Scvb0 *ctx, int *doc_word, int n_word, int max_iter);
float *scvb0FitTransform(Scvb0 *ctx, int** word_indexes_ptr, short** word_counts_ptr, int* n_word_each_doc, int* n_word_type_each_doc,long long n_all_word, int n_document, int n_word_type);
void scvb0Free(Scvb0* ctx);

void* dmalloc(size_t size);
void* dcalloc(size_t numb, size_t size);


#ifdef __cplusplus
}
#endif

#endif /* __CHASEN_H__ */
