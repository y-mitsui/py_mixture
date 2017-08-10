CFLAGS=-g -O2
LOADLIBES=-lpthread
all:
	gcc -std=c99 -O2 -c -fPIC lda_scvb0_thread.c -o lda_scvb0.o
	gcc -std=c99 -O2  -shared -fPIC -I${JNI_HEADER} -I${JNI_HEADER2} LdaScvb0Wrap.c lda_scvb0.o -o libLdaScvb0Wrap.so -lpthread

lda_scvb0:
lda_scvb0_thread: update_lda.o
