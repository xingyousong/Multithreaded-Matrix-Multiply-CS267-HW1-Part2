#include <x86intrin.h>
#include <immintrin.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#if !defined(BLOCK_SIZE_LARGE)
#define BLOCK_SIZE_LARGE 128
#endif

#if !defined(BLOCK_SIZE_SMALL)
#define BLOCK_SIZE_SMALL 64
#endif

#if !defined(REGISTER_SIZE)
#define REGISTER_SIZE 4
#endif

#if !defined(STRIDE)
#define STRIDE 8
#endif


#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))
#define row_major_offset(i, j, lda) (i * lda + j)
#define col_major_offset(i, j, lda) (j * lda + i)
#define weird_offset(i, j, lda, stride) (i*lda + j*stride)
#define weird_offset_no_multiply(i, j, lda, stride) ((i/STRIDE) * STRIDE * lda + j * STRIDE + i % STRIDE)
 
const char* dgemm_desc = "Yao Fixed WIP (Hopefully not RIP) blocked dgemm.";


static double* weird_transformation(double* src, int lda, int stride) {
  // //we would like to have the array been divided into multiple subarrays
  // //The number of columns should be a multiply of stride
  int rowNums = ceil( (double)lda / stride);
  int colNums = lda * stride;

  double* dest __attribute__((aligned(32))) = malloc(rowNums * colNums * sizeof(double));

  for (int i = 0; i < lda * lda; i++){
    int whichRow = (i % lda) / stride;
    int whichCol = i / lda * stride + i % lda - whichRow * stride;
    dest[whichRow * colNums + whichCol] = src[i];
  }
  return dest;
}


// A   M * K
// B   K * N
// C   M * N
static void compute(double* A, double* B, double* C, int M, int N, int K, int lda){
  __assume_aligned(A, 32);
  __assume_aligned(B, 32);
  __assume_aligned(C, 32);

  __m256d c_col_0, c_col_1, c_col_2, c_col_3, c_col_4, c_col_5, c_col_6, c_col_7;
  __m256d b_k0, b_k1, b_k2, b_k3;
  
  double*  __attribute__((aligned(32))) C_LU = C;

  for (int i = 0; i <= M - 8; i += 8){
    for (int j = 0; j <= N - 4; j += 4){
        double* Cij = C_LU + j*lda;
        //load cols
        c_col_0 = _mm256_load_pd(Cij);
        c_col_1 = _mm256_load_pd(Cij + lda);
        c_col_2 = _mm256_load_pd(Cij + 2*lda);
        c_col_3 = _mm256_load_pd(Cij + 3*lda);
        c_col_4 = _mm256_load_pd(Cij + 4);
        c_col_5 = _mm256_load_pd(Cij + lda + 4);
        c_col_6 = _mm256_load_pd(Cij + 2*lda + 4);
        c_col_7 = _mm256_load_pd(Cij + 3*lda + 4);
     
        __m256d a_row_k_first_half;
        __m256d a_row_k_second_half;
        for (int k = 0; k < K; ++k){
          a_row_k_first_half = _mm256_load_pd(A+weird_offset(i,k,lda,STRIDE));
          a_row_k_second_half = _mm256_load_pd(A+weird_offset(i,k,lda,STRIDE)+ 4);

          //broadcast might be faster
          b_k0 = _mm256_set1_pd(B[k+j*lda]);
          b_k1 = _mm256_set1_pd(B[k+(j+1)*lda]);
          b_k2 = _mm256_set1_pd(B[k+(j+2)*lda]);
          b_k3 = _mm256_set1_pd(B[k+(j+3)*lda]);

          c_col_0 = _mm256_fmadd_pd(a_row_k_first_half, b_k0, c_col_0);     
          c_col_1 = _mm256_fmadd_pd(a_row_k_first_half, b_k1, c_col_1);
          c_col_2 = _mm256_fmadd_pd(a_row_k_first_half, b_k2, c_col_2);
          c_col_3 = _mm256_fmadd_pd(a_row_k_first_half, b_k3, c_col_3);
          c_col_4 = _mm256_fmadd_pd(a_row_k_second_half, b_k0, c_col_4);     
          c_col_5 = _mm256_fmadd_pd(a_row_k_second_half, b_k1, c_col_5);
          c_col_6 = _mm256_fmadd_pd(a_row_k_second_half, b_k2, c_col_6);
          c_col_7 = _mm256_fmadd_pd(a_row_k_second_half, b_k3, c_col_7);


        }
        _mm256_store_pd(Cij,         c_col_0);
        _mm256_store_pd(Cij+lda,     c_col_1);
        _mm256_store_pd(Cij+2*lda,   c_col_2);
        _mm256_store_pd(Cij+3*lda,   c_col_3);
        _mm256_store_pd(Cij+4,       c_col_4);
        _mm256_store_pd(Cij+lda+4,   c_col_5);
        _mm256_store_pd(Cij+2*lda+4, c_col_6);
        _mm256_store_pd(Cij+3*lda+4, c_col_7);
    }

    //leftover//
    for (int j = (N/4)*4; j < N; j++){
        c_col_0 = _mm256_load_pd(C + i + j*lda);
        c_col_1 = _mm256_load_pd(C + i + j*lda + 4);
        __m256d a_row_k_first_half;
        __m256d a_row_k_second_half;
        for (int k = 0; k < K; k++){
          a_row_k_first_half  = _mm256_load_pd(A+weird_offset(i,k,lda,STRIDE));
          a_row_k_second_half = _mm256_load_pd(A+weird_offset(i,k,lda,STRIDE)+4);
          b_k0                = _mm256_broadcast_sd(B+k+j*lda);
          c_col_0             = _mm256_fmadd_pd(a_row_k_first_half, b_k0, c_col_0);
          c_col_1             = _mm256_fmadd_pd(a_row_k_second_half, b_k0, c_col_1);
        }
        _mm256_store_pd(C+i+j*lda,      c_col_0);
        _mm256_store_pd(C+i+j*lda + 4,  c_col_1);
    }
    C_LU += STRIDE;
  }

  for (int i = (M/8)*8; i <= M - 4; i += 4){
    for (int j = 0; j <= N - 4; j += 4){
        double* Cij = C_LU + j*lda;
        //load cols
        c_col_0 = _mm256_load_pd(Cij);
        c_col_1 = _mm256_load_pd(Cij + lda);
        c_col_2 = _mm256_load_pd(Cij + 2*lda);
        c_col_3 = _mm256_load_pd(Cij + 3*lda);

        __m256d a_row_k_first_half;
        for (int k = 0; k < K; ++k){
          a_row_k_first_half = _mm256_load_pd(A+weird_offset_no_multiply(i, k, lda, STRIDE));

          //broadcast might be faster
          b_k0 = _mm256_set1_pd(B[k+j*lda]);
          b_k1 = _mm256_set1_pd(B[k+(j+1)*lda]);
          b_k2 = _mm256_set1_pd(B[k+(j+2)*lda]);
          b_k3 = _mm256_set1_pd(B[k+(j+3)*lda]);

          c_col_0 = _mm256_fmadd_pd(a_row_k_first_half, b_k0, c_col_0);     
          c_col_1 = _mm256_fmadd_pd(a_row_k_first_half, b_k1, c_col_1);
          c_col_2 = _mm256_fmadd_pd(a_row_k_first_half, b_k2, c_col_2);
          c_col_3 = _mm256_fmadd_pd(a_row_k_first_half, b_k3, c_col_3);
        }
        _mm256_store_pd(Cij,         c_col_0);
        _mm256_store_pd(Cij+lda,     c_col_1);
        _mm256_store_pd(Cij+2*lda,   c_col_2);
        _mm256_store_pd(Cij+3*lda,   c_col_3);
    }

    //leftover//
    for (int j = (N/4)*4; j < N; j++){
        c_col_0 = _mm256_load_pd(C + i + j*lda);
        __m256d a_row_k_first_half;
        for (int k = 0; k < K; k++){
          a_row_k_first_half  = _mm256_load_pd(A+weird_offset(i,k,lda,STRIDE));
          b_k0                = _mm256_broadcast_sd(B+k+j*lda);
          c_col_0             = _mm256_fmadd_pd(a_row_k_first_half, b_k0, c_col_0);
        }
        _mm256_store_pd(C+i+j*lda,      c_col_0);
    }
    C_LU += 4;
  }

  //Ultimate Leftover, Brute Force 
  for (int i = (M/4)*4; i < M; ++i){
      for (int j = 0; j < N; ++j){
          double C_ij = C[i+j*lda];
          for (int k = 0; k < K; k++){
            C_ij += A[weird_offset_no_multiply(i, k, lda, STRIDE)] * B[k+j*lda];
          }
          C[i+j*lda] = C_ij;
      }
  }
}



// A   M * K
// B   K * N
// C   M * N
static void do_block_large(double* A, double* B, double* C, int M, int N, int K, int lda){
  
  //tweak size
  int SMALL_M = 128;
  int SMALL_N = 128;
  int SMALL_K = 128;

  //tweak loop
  for (int k = 0; k < K; k += SMALL_K){
    int K_ = min (SMALL_K, K-k);

    for (int j = 0; j < N; j += SMALL_N){
      int N_ = min (SMALL_N, N-j);

      for (int i = 0; i < M; i += SMALL_M){
        int M_ = min (SMALL_M, M-i);
        
        compute(A + weird_offset(i, k, lda, STRIDE), B+col_major_offset(k, j, lda), C+col_major_offset(i, j, lda), M_, N_, K_, lda);
      }
    }
  }
}

void square_dgemm (int lda, double* A, double* B, double* C){
  double* weird_A = weird_transformation(A, lda, STRIDE);

  //tweak size
  int LARGE_M = 128;
  int LARGE_N = 256;
  int LARGE_K = 512;

  //tweak loop
  for (int i = 0; i < lda; i += LARGE_M){
    int M = min (LARGE_M, lda-i);

    for (int j = 0; j < lda; j += LARGE_N){
      int N = min (LARGE_N, lda-j);

      for (int k = 0; k < lda; k += LARGE_K){
        int K = min (LARGE_K, lda-k);
        
        do_block_large(weird_A + weird_offset(i, k, lda, STRIDE), B+col_major_offset(k, j, lda), C+col_major_offset(i, j, lda), M, N, K, lda);
      }
    }
  }
  free(weird_A);
}
