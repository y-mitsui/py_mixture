#include "com_teamlab_selectware_learnings_JniLdaScvb0.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lda.h"

JNIEXPORT void JNICALL Java_com_teamlab_selectware_learnings_JniLdaScvb0_fit
	(JNIEnv *env, jobject thisj, jobjectArray word_indexes, jobjectArray word_counter, jobjectArray result_topics) {
	jclass this_clsj = (*env)->GetObjectClass(env, thisj);
	jfieldID fid = (*env)->GetFieldID(env, this_clsj, "n_components" , "I");
	int n_components = (*env)->GetIntField(env, thisj, fid );
	fid = (*env)->GetFieldID(env, this_clsj, "n_iter" , "I");
	int n_iter = (*env)->GetIntField(env, thisj, fid );
	fid = (*env)->GetFieldID(env, this_clsj, "batch_size" , "I");
	int batch_size = (*env)->GetIntField(env, thisj, fid );
	fid = (*env)->GetFieldID(env, this_clsj, "n_thread" , "I");
	int n_thread = (*env)->GetIntField(env, thisj, fid );

	int n_row = (*env)->GetArrayLength(env, word_indexes);
	int **word_indexes_wrap = (int **)dmalloc(sizeof(int*) * n_row);
	short **word_counter_wrap = ( short **)dmalloc(sizeof( short*) * n_row);
	int *n_word_each_doc = (int *)dcalloc(1, sizeof(int) * n_row);
	int *n_word_type_each_doc = (int *)dmalloc(sizeof(int) * n_row);
	int n_all_word = 0;
	int n_all_word_type = 0;
	for(int i=0; i < n_row; i++){
		jintArray indexes_array = (jintArray)(*env)->GetObjectArrayElement(env, word_indexes, i);
		jintArray counter_array = (jintArray)(*env)->GetObjectArrayElement(env, word_counter, i);
		if(indexes_array == NULL) {
			n_word_each_doc[i] = 0;
			n_word_type_each_doc[i] = 0;
			word_indexes_wrap[i] = NULL;
			word_counter_wrap[i] = NULL;
			continue;
		}
		jint *indexes_element = (*env)->GetIntArrayElements(env, indexes_array, 0);
		jint *counter_element = (*env)->GetIntArrayElements(env, counter_array, 0);
		int n_col = (*env)->GetArrayLength(env, indexes_array);
		word_indexes_wrap[i] =(int*)dmalloc(sizeof(int) * n_col);
		word_counter_wrap[i] =(short*)dmalloc(sizeof(short) * n_col);
		n_word_type_each_doc[i] = n_col;
		for (int j=0; j < n_col; j++) {
			word_indexes_wrap[i][j] = indexes_element[j];
			word_counter_wrap[i][j] = counter_element[j];
			n_word_each_doc[i] += counter_element[j];
		}
		n_all_word += n_word_each_doc[i];
		n_all_word_type += n_word_type_each_doc[i];
		(*env)->ReleaseIntArrayElements(env, indexes_array, indexes_element, 0);
		(*env)->ReleaseIntArrayElements(env, counter_array, counter_element, 0);
		(*env)->DeleteLocalRef(env, indexes_array);
		(*env)->DeleteLocalRef(env, counter_array);
	}

	float *topics = dmalloc(sizeof(double) * n_row * n_components);
	Scvb0* sctx = scvb0Init(n_components, n_iter, batch_size, n_thread, 1.0, 1.0);
	scvb0Fit(sctx, word_indexes_wrap, word_counter_wrap, n_word_each_doc, n_word_type_each_doc, n_all_word, n_row, n_all_word_type, topics);
	for (int i=0; i < n_row; i++) {
		jdoubleArray result_array = (jintArray)(*env)->GetObjectArrayElement(env, result_topics, i);
		jdouble *result_element = (*env)->GetDoubleArrayElements(env, result_array, 0);
		for (int j=0; j < n_components; j++) {
			result_element[j] = (double)topics[i * n_components + j];
		}
		(*env)->ReleaseDoubleArrayElements(env, result_array, result_element, 0);
		(*env)->DeleteLocalRef(env, result_array);
	}
	for(int i=0; i < n_row; i++){
		if (word_indexes_wrap[i] != NULL) {
			free(word_indexes_wrap[i]);
			free(word_counter_wrap[i]);
		}
	}
	free(n_word_each_doc);
	free(n_word_type_each_doc);
	free(word_indexes_wrap);
	free(word_counter_wrap);
}
