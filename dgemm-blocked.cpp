#include "vectorclass.h"
#include <iostream>

#ifdef __AVX__
#include <immintrin.h>
#else
#warning AVX is not available. Code will not compile!
#endif

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 32
#endif

#define min(a, b) (((a)<(b))?(a):(b))

extern "C" {
const char *dgemm_desc = "Simple blocked dgemm.";


void padMatrix(double *padX, double *X, int lda, int lda1) {
    for (int j = 0; j < lda; j++) {
        for (int i = 0; i < lda; i++) {
            padX[i + (j * lda1)] = X[i + (j * lda)];
        }
    }
}


void unpadMatrix(double *padX, double *X, int lda, int lda1) {
    for (int j = 0; j < lda; j++) {
        for (int i = 0; i < lda; i++) {
            X[i + (j * lda)] = padX[i + (j * lda1)];
        }
    }
}

static void do_block(int lda, int M, int N, int K, double *A, double *B, double *C) {

    Vec4d c[8], a[2], b[4];
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < N; j += 4) {
            b[0] = B[k + (j * lda)];
            b[1] = (j + 1 < N) ? B[k + ((j + 1) * lda)] : 0.0;
            b[2] = (j + 2 < N) ? B[k + ((j + 2) * lda)] : 0.0;
            b[3] = (j + 3 < N) ? B[k + ((j + 3) * lda)] : 0.0;
            for (int i = 0; i < M; i += 8) {
                a[0].load_partial(min(4, M - i), &A[i + (k * lda)]);
                a[1].load_partial(min(4, M - i - 4), &A[i + 4 + (k * lda)]);

                c[0].load_partial(min(4, M - i), &C[i + (j * lda)]);
                c[1].load_partial(min(4, (j + 1 < N) ? M - i : 0), &C[i + ((j + 1) * lda)]);
                c[2].load_partial(min(4, (j + 2 < N) ? M - i : 0), &C[i + ((j + 2) * lda)]);
                c[3].load_partial(min(4, (j + 3 < N) ? M - i : 0), &C[i + ((j + 3) * lda)]);
                c[4].load_partial(min(4, M - i - 4), &C[i + 4 + (j * lda)]);
                c[5].load_partial(min(4, (j + 1 < N) ? M - i - 4 : 0), &C[i + 4 + ((j + 1) * lda)]);
                c[6].load_partial(min(4, (j + 2 < N) ? M - i - 4 : 0), &C[i + 4 + ((j + 2) * lda)]);
                c[7].load_partial(min(4, (j + 3 < N) ? M - i - 4 : 0), &C[i + 4 + ((j + 3) * lda)]);


                c[0] = mul_add(a[0], b[0], c[0]);
                c[1] = mul_add(a[0], b[1], c[1]);
                c[2] = mul_add(a[0], b[2], c[2]);
                c[3] = mul_add(a[0], b[3], c[3]);
                c[4] = mul_add(a[1], b[0], c[4]);
                c[5] = mul_add(a[1], b[1], c[5]);
                c[6] = mul_add(a[1], b[2], c[6]);
                c[7] = mul_add(a[1], b[3], c[7]);

                c[0].store_partial(min(4, M - i), &C[i + (j * lda)]);
                c[1].store_partial(min(4, (j + 1 < N) ? M - i : 0), &C[i + ((j + 1) * lda)]);
                c[2].store_partial(min(4, (j + 2 < N) ? M - i : 0), &C[i + ((j + 2) * lda)]);
                c[3].store_partial(min(4, (j + 3 < N) ? M - i : 0), &C[i + ((j + 3) * lda)]);
                c[4].store_partial(min(4, M - i - 4), &C[i + 4 + (j * lda)]);
                c[5].store_partial(min(4, (j + 1 < N) ? M - i - 4 : 0), &C[i + 4 + ((j + 1) * lda)]);
                c[6].store_partial(min(4, (j + 2 < N) ? M - i - 4 : 0), &C[i + 4 + ((j + 2) * lda)]);
                c[7].store_partial(min(4, (j + 3 < N) ? M - i - 4 : 0), &C[i + 4 + ((j + 3) * lda)]);
            }
        }
    }
}

void square_dgemm(int lda, double *A, double *B, double *C) {
//    int lda1 = lda;
//    //calculate the additional number of elements to be added if the number of elements by row is not a multiple of 8
//    if (lda % 8 != 0) {
//        int modval = lda % 8;
//        lda1 = lda + (8 - modval) + 8;
//    }
//
//    double *X = (double *) _mm_malloc(lda1 * lda1 * sizeof(double), 32);
//    double *Y = (double *) _mm_malloc(lda1 * lda1 * sizeof(double), 32);
//    double *Z = (double *) _mm_malloc(lda1 * lda1 * sizeof(double), 32);
//    padMatrix(X, A, lda, lda1);
//    padMatrix(Y, B, lda, lda1);
//    padMatrix(Z, C, lda, lda1);


    // For each block-row of A
    for (int i = 0; i < lda; i += BLOCK_SIZE) {
        //  For each block-column of B
        for (int j = 0; j < lda; j += BLOCK_SIZE) {

            // Accumulate block dgemms into block of C
            for (int k = 0; k < lda; k += BLOCK_SIZE) {
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min (BLOCK_SIZE, lda - i);
                int N = min (BLOCK_SIZE, lda - j);
                int K = min (BLOCK_SIZE, lda - k);

                // Perform individual block dgemm
                do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
            }
        }
    }
//    unpadMatrix(Z, C, lda, lda1);
//    _mm_free(X);
//    _mm_free(Y);


}

}
