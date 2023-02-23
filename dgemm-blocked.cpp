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

//
//
//void dgemm(int m, int n, int k, double* A, double* B, double* C) {
//    const int vec_size = 4; // VCL uses 128-bit vectors (4 doubles)
//    const int vec_size = vec_size / sizeof(double);
//
//    // Create VCL vectors for A, B, and C
//    vcl::vec<double, vec_size> vA[k][m / vec_size];
//    vcl::vec<double, vec_size> vB[n / vec_size][k];
//    vcl::vec<double, vec_size> vC[n / vec_size][m / vec_size];
//
//    // Load A and B into VCL vectors
//    for (int i = 0; i < k; i++) {
//        for (int j = 0; j < m / vec_size; j++) {
//            vA[i][j].load(&A[i * m + j * vec_size]);
//        }
//    }
//    for (int i = 0; i < n / vec_size; i++) {
//        for (int j = 0; j < k; j++) {
//            vB[i][j].load(&B[j * n + i * vec_size]);
//        }
//    }
//
//    // Compute C using dot products
//    for (int i = 0; i < n / vec_size; i++) {
//        for (int j = 0; j < m / vec_size; j++) {
//            vC[i][j] = 0;
//            for (int l = 0; l < k; l++) {
//                vC[i][j] += vB[i][l] * vA[l][j];
//            }
//        }
//    }
//
//    // Convert VCL vectors back to C arrays
//    for (int i = 0; i < n / vec_size; i++) {
//        for (int j = 0; j < m / vec_size; j++) {
//            vC[i][j].store(&C[i * m + j * vec_size]);
//        }
//    }
//}

static void do_block(int lda, int M, int N, int K, double *A, double *B, double *C) {

    Vec4d c[8], a[2], b[4];
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < N; j += 4) {
            b[0] = B[k + (j * lda)];
            b[1] = B[k + ((j + 1) * lda)];
            b[2] = B[k + ((j + 2) * lda)];
            b[3] = B[k + ((j + 3) * lda)];
            for (int i = 0; i < M; i += 8) {

                a[0].load(&A[i + (k * lda)]);
                a[1].load(&A[i + 4 + (k * lda)]);

                c[0].load(&C[i + (j * lda)]);
                c[1].load(&C[i + ((j + 1) * lda)]);
                c[2].load(&C[i + ((j + 2) * lda)]);
                c[3].load(&C[i + ((j + 3) * lda)]);
                c[4].load(&C[i + 4 + (j * lda)]);
                c[5].load(&C[i + 4 + ((j + 1) * lda)]);
                c[6].load(&C[i + 4 + ((j + 2) * lda)]);
                c[7].load(&C[i + 4 + ((j + 3) * lda)]);


                c[0] = mul_add(a[0], b[0], c[0]);
                c[1] = mul_add(a[0], b[1], c[1]);
                c[2] = mul_add(a[0], b[2], c[2]);
                c[3] = mul_add(a[0], b[3], c[3]);
                c[4] = mul_add(a[1], b[0], c[4]);
                c[5] = mul_add(a[1], b[1], c[5]);
                c[6] = mul_add(a[1], b[2], c[6]);
                c[7] = mul_add(a[1], b[3], c[7]);

                c[0].store(&C[i + (j * lda)]);
                c[1].store(&C[i + ((j + 1) * lda)]);
                c[2].store(&C[i + ((j + 2) * lda)]);
                c[3].store(&C[i + ((j + 3) * lda)]);
                c[4].store(&C[i + 4 + (j * lda)]);
                c[5].store(&C[i + 4 + ((j + 1) * lda)]);
                c[6].store(&C[i + 4 + ((j + 2) * lda)]);
                c[7].store(&C[i + 4 + ((j + 3) * lda)]);
            }
        }
    }
}

void square_dgemm(int lda, double *A, double *B, double *C) {
    int lda1 = lda;
    //calculate the additional number of elements to be added if the number of elements by row is not a multiple of 8
    if (lda % 8 != 0) {
        int modval = lda % 8;
        lda1 = lda + (8 - modval) + 8;
    }

    double *X = (double *) _mm_malloc(lda1 * lda1 * sizeof(double), 32);
    double *Y = (double *) _mm_malloc(lda1 * lda1 * sizeof(double), 32);
    double *Z = (double *) _mm_malloc(lda1 * lda1 * sizeof(double), 32);
    padMatrix(X, A, lda, lda1);
    padMatrix(Y, B, lda, lda1);
    padMatrix(Z, C, lda, lda1);


    // For each block-row of A
    for (int i = 0; i < lda1; i += BLOCK_SIZE) {
        //  For each block-column of B
        for (int j = 0; j < lda1; j += BLOCK_SIZE) {

            // Accumulate block dgemms into block of C
            for (int k = 0; k < lda1; k += BLOCK_SIZE) {
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min (BLOCK_SIZE, lda1 - i);
                int N = min (BLOCK_SIZE, lda1 - j);
                int K = min (BLOCK_SIZE, lda1 - k);

                // Perform individual block dgemm
                do_block(lda1, M, N, K, X + i + k * lda1, Y + k + j * lda1, Z + i + j * lda1);
            }
        }
    }
    unpadMatrix(Z, C, lda, lda1);
    _mm_free(X);
    _mm_free(Y);


}

}
