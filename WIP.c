#include <x86intrin.h>
#include <immintrin.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

#if !defined(STRIDE)
#define STRIDE 8
#endif

double sec_1 = -1;
double sec_2 = -1;


#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))
#define row_major_offset(i, j, lda) (i * lda + j)
#define col_major_offset(i, j, lda) (j * lda + i)
#define weird_offset(i, j, lda, stride) (i*lda + j*stride)
#define weird_offset_no_multiply(i, j, lda, stride) ((i/STRIDE) * STRIDE * lda + j * STRIDE + i % STRIDE)
 
const char* dgemm_desc = "Yao & Xingyou's Optimized blocked dgemm.";


static double* weird_transformation(double* src, int lda, int stride) {
  // //we would like to have the array been divided into multiple subarrays
  // //The number of columns should be a multiply of stride
  int rowNums = ceil( (double)lda / stride);
  int colNums = lda * stride;

  double* dest __attribute__((aligned(32))) = malloc(rowNums * colNums * sizeof(double));

  #pragma omp parallel num_threads(16)
  {
    #pragma omp for //divide the whole for loop by two threads
    for (int i = 0; i < lda * lda; i++){
      dest[i] = 0;
    }
  }

  #pragma omp parallel num_threads(16)
  {
    #pragma omp for //divide the whole for loop by two threads
    for (int i = 0; i < lda * lda; i++){
      int whichRow = (i % lda) / stride;
      int whichCol = i / lda * stride + i % lda - whichRow * stride;
      dest[whichRow * colNums + whichCol] = src[i];
    }
  }
  return dest;
}

static double* super_padding(double* src, int lda, int stride){
  int padded_lda = ceil((double)lda / stride) * stride;
  double* dest __attribute__((aligned(32))) = malloc(padded_lda * padded_lda * sizeof(double));

  #pragma omp parallel num_threads(32)
  {
    #pragma omp for //divide the whole for loop by two threads
    for (int i = 0; i < padded_lda* padded_lda; i++){
      dest[i] = 0;
    }
  }

  #pragma omp parallel num_threads(32)
  {
    #pragma omp for
    for (int i = 0; i < lda; i++){
      for (int j = 0; j < lda; j++){
        dest[i + j * padded_lda] = src[i + j * lda];
      }
    }
  }
  return dest;
}

static void store_back(double* dest, double* src, int lda, int padded_lda){
  #pragma omp parallel num_threads(32)
  {
    #pragma omp for
    for (int i = 0; i < lda; i++){
      for (int j = 0; j < lda; j++){
        src[i + j * lda] = dest[i + j * padded_lda];
      }
    }
  }
}


//Handle 4 * 8
static void compute_four_by_eight(double* A, double* B, double* C, int M, int N, int K, int lda){
  __assume_aligned(A, 32);
  __assume_aligned(B, 32);
  __assume_aligned(C, 32);

  __m256d c_col_0, c_col_1, c_col_2, c_col_3, c_col_4, c_col_5, c_col_6, c_col_7;
  __m256d b_k0, b_k1, b_k2, b_k3;

  printf("M %d, N %d, K %d, lda %d \n", M, N, K, lda);

  #pragma omp parallel num_threads(2)
  {
    #pragma omp for //divide the whole for loop by two threads

    for (int i = 0; i <= M - STRIDE; i += STRIDE){
      for (int j = 0; j <= N - 4; j += 4){
          double* Cij = C + i + j*lda;
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
    }
  }
}

//Handle 4 * 4
static void compute_four_by_four(double* A, double* B, double* C, int M, int N, int K, int lda){
  __m256d c_col_0, c_col_1, c_col_2, c_col_3;
  __m256d b_k0, b_k1, b_k2, b_k3;

  #pragma omp parallel num_threads(1)
  {
    #pragma omp for //divide the whole for loop by two threads
    for (int i = (M/STRIDE)*STRIDE; i <= M-4; i += 4){

      #pragma omp parallel num_threads(2)
      #pragma omp for 
      for (int j = 0; j <= N - 4; j += 4){
          double* Cij = C + i + j*lda;
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
    }
  }
}

//naive brutal force
static void naive(double* A, double* B, double* C, int M, int N, int K, int lda){
  __assume_aligned(A, 32);
  __assume_aligned(B, 32);
  __assume_aligned(C, 32);

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
static void compute(double* A, double* B, double* C, int M, int N, int K, int lda){
  //The following methods can be called in arbitrary sequence.

  #pragma omp parallel sections num_threads(1)
  {    
    
    #pragma omp section
    {
      //first handle edge case using naive
      //The lightest section
      double tdata_0 = omp_get_wtime();
      //naive(A, B, C, M, N, K, lda);
      tdata_0 = omp_get_wtime() - tdata_0;
      //printf("lda: %d, Section 0: %f secs\n", lda, tdata_0);
      
      //then handle those cannot be computed using 4 by 8
      //Second Lightest section
      double tdata_1 = omp_get_wtime();
      //compute_four_by_four(A, B, C, M, N, K, lda);
      tdata_1 = omp_get_wtime() - tdata_1;
      //printf("lda: %d, Section 1: %f secs\n", lda, tdata_1);
    }
    
    #pragma omp section
    {
      //lastly handle those can be computed using 4 by 8
      //The heaviest section
      double tdata_2 = omp_get_wtime();
      compute_four_by_eight(A, B, C, M, N, K, lda);
      tdata_2 = omp_get_wtime() - tdata_2;
      //printf("lda: %d, Section 2: %f secs\n", lda, tdata_2);
    }
  }
}



// A   M * K
// B   K * N
// C   M * N
static void divide_into_small_blocks(double* A, double* B, double* C, int M, int N, int K, int lda){
  
  //tweak size
  // int SMALL_M = 32;
  // int SMALL_N = 64;
  // int SMALL_K = 128;

  int SMALL_M = 32;
  int SMALL_N = 64;
  int SMALL_K = 128;

  //tweak loop
  #pragma omp parallel num_threads(1)
  {
    for (int k = 0; k < K; k += SMALL_K){
      int sub_K = min (SMALL_K, K-k);
      
      #pragma omp parallel num_threads(max(2, min(16, (M / SMALL_M + 1))))
      {
        #pragma omp for
        for (int i = 0; i < M; i += SMALL_M){
          int sub_M = min (SMALL_M, M-i);

          #pragma omp parallel num_threads(1)
          {
            #pragma omp for
            for (int j = 0; j < N; j += SMALL_N){
              int sub_N = min (SMALL_N, N-j);
              
              compute(A + weird_offset(i, k, lda, STRIDE), B+col_major_offset(k, j, lda), C+col_major_offset(i, j, lda), sub_M, sub_N, sub_K, lda);
            }
          }
        }
      }
    }
  }
}

static void divide_into_large_blocks(double* A, double* B, double* C, int lda){

  double x = omp_get_wtime();

  double* weird_A = weird_transformation(A, lda, STRIDE);

  x = omp_get_wtime() - x;
  
  //tweak size
  int LARGE_M = 2048;
  int LARGE_N = 256;
  int LARGE_K = 512;

  //tweak loop
  for (int i = 0; i < lda; i += LARGE_M){
    int M = min (LARGE_M, lda-i);

    for (int j = 0; j < lda; j += LARGE_N){
      int N = min (LARGE_N, lda-j);

      for (int k = 0; k < lda; k += LARGE_K){
        int K = min (LARGE_K, lda-k);
        
        divide_into_small_blocks(weird_A + weird_offset(i, k, lda, STRIDE), B+col_major_offset(k, j, lda), C+col_major_offset(i, j, lda), M, N, K, lda);
      }
    }
  }
  free(weird_A);
}

void square_dgemm (int lda, double* A, double* B, double* C){

  int padded_lda = ceil((double)lda / STRIDE) * STRIDE;

  double* padded_A __attribute__((aligned(32))) = super_padding(A, lda, STRIDE);
  double* padded_B __attribute__((aligned(32))) = super_padding(B, lda, STRIDE);
  double* padded_C __attribute__((aligned(32))) = super_padding(C, lda, STRIDE);


  divide_into_large_blocks(padded_A, padded_B, padded_C, padded_lda);

  store_back(C, padded_C, lda, padded_lda);

  free(padded_A);
  free(padded_B);
  free(padded_C);
}