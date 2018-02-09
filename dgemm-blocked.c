#include <x86intrin.h>
#include <immintrin.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

#if !defined(STRIDE)
#define STRIDE 8
#endif


#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))
#define row_major_offset(i, j, lda) (i * lda + j)
#define col_major_offset(i, j, lda) (j * lda + i)
#define weird_offset(i, j, lda, stride) (i*lda + j*stride)
#define weird_offset_no_multiply(i, j, lda, stride) ((i/STRIDE) * STRIDE * lda + j * STRIDE + i % STRIDE)
 
const char* dgemm_desc = "Yao & Xingyou's Optimized blocked dgemm v0.2.";


static double* weird_transformation(double* src, int lda, int stride) {
  // //we would like to have the array been divided into multiple subarrays
  // //The number of columns should be a multiply of stride
  int rowNums = ceil( (double)lda / stride);
  int colNums = lda * stride;

  double* dest __attribute__((aligned(32))) = malloc(rowNums * colNums * sizeof(double));

  #pragma omp parallel num_threads(min(32, (lda/stride)))
  {
    #pragma omp for //divide the whole for loop by two threads
    for (int i = 0; i < lda; i++){
      int i_pos = weird_offset_no_multiply(i, 0, lda, stride);
      for (int j = 0; j < lda; j++){
        dest[i_pos + j * STRIDE] = src[col_major_offset(i, j, lda)];
      }
    }
  }
  return dest;
}


//Handle 4 * 8
static void compute_four_by_eight(double* A, double* B, double* C, int M, int N, int K, int lda){
  __assume_aligned(A, 32);
  __assume_aligned(B, 32);
  __assume_aligned(C, 32);

  __m256d c_col_0, c_col_1, c_col_2, c_col_3, c_col_4, c_col_5, c_col_6, c_col_7;
  __m256d b_k0, b_k1, b_k2, b_k3;

  #pragma omp parallel num_threads(1)
  {
    #pragma omp for //divide the whole for loop by two threads

    for (int i = 0; i <= M - STRIDE; i += STRIDE){
      for (int j = 0; j <= N - 4; j += 4){
          double* C_ij = C + i + j*lda;
          //load cols
          c_col_0 = _mm256_load_pd(C_ij);
          c_col_1 = _mm256_load_pd(C_ij + lda);
          c_col_2 = _mm256_load_pd(C_ij + 2*lda);
          c_col_3 = _mm256_load_pd(C_ij + 3*lda);
          c_col_4 = _mm256_load_pd(C_ij + 4);
          c_col_5 = _mm256_load_pd(C_ij + lda + 4);
          c_col_6 = _mm256_load_pd(C_ij + 2*lda + 4);
          c_col_7 = _mm256_load_pd(C_ij + 3*lda + 4);
       
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
          _mm256_store_pd(C_ij,         c_col_0);
          _mm256_store_pd(C_ij+lda,     c_col_1);
          _mm256_store_pd(C_ij+2*lda,   c_col_2);
          _mm256_store_pd(C_ij+3*lda,   c_col_3);
          _mm256_store_pd(C_ij+4,       c_col_4);
          _mm256_store_pd(C_ij+lda+4,   c_col_5);
          _mm256_store_pd(C_ij+2*lda+4, c_col_6);
          _mm256_store_pd(C_ij+3*lda+4, c_col_7);
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
    for (int i = (M/STRIDE)*STRIDE; i <= M - 4; i += 4){
      #pragma omp parallel num_threads(1)
      {
        #pragma omp for
        for (int j = 0; j <= N - 4; j += 4){
            double* C_ij = C + i + j*lda;
            //load cols
            c_col_0 = _mm256_load_pd(C_ij);
            c_col_1 = _mm256_load_pd(C_ij + lda);
            c_col_2 = _mm256_load_pd(C_ij + 2*lda);
            c_col_3 = _mm256_load_pd(C_ij + 3*lda);

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
            _mm256_store_pd(C_ij,         c_col_0);
            _mm256_store_pd(C_ij+lda,     c_col_1);
            _mm256_store_pd(C_ij+2*lda,   c_col_2);
            _mm256_store_pd(C_ij+3*lda,   c_col_3);
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
}

//naive brutal force
static void two_by_two_and_naive(double* A, double* B, double* C, int M, int N, int K, int lda){
  __assume_aligned(A, 32);
  __assume_aligned(B, 32);
  __assume_aligned(C, 32);

  __m128d c_col_0, c_col_1, c_col_2, c_col_3;
  __m128d b_k0, b_k1, b_k2, b_k3;
  for (int i = (M/4)*4; i <= M-2; i+=2){
    #pragma omp parallel num_threads(1)
    {
      #pragma omp for
      for (int j = 0; j <= N-4; j += 4){
        double* C_ij = C + i + j*lda;
        //load cols
        c_col_0 = _mm_load_pd(C_ij);
        c_col_1 = _mm_load_pd(C_ij + lda);
        c_col_2 = _mm_load_pd(C_ij + 2 * lda);
        c_col_3 = _mm_load_pd(C_ij + 3 * lda);

        __m128d a_row_k_first_half;
        int A_pos = weird_offset_no_multiply(i, 0, lda, STRIDE);
        for (int k = 0; k < K; ++k){
          a_row_k_first_half = _mm_load_pd(A + A_pos + k * STRIDE);

          //broadcast might be faster
          b_k0 = _mm_set1_pd(B[k+j*lda]);
          b_k1 = _mm_set1_pd(B[k+(j+1)*lda]);
          b_k2 = _mm_set1_pd(B[k+(j+2)*lda]);
          b_k3 = _mm_set1_pd(B[k+(j+3)*lda]);

          c_col_0 = _mm_fmadd_pd(a_row_k_first_half, b_k0, c_col_0);
          c_col_1 = _mm_fmadd_pd(a_row_k_first_half, b_k1, c_col_1);
          c_col_2 = _mm_fmadd_pd(a_row_k_first_half, b_k2, c_col_2);
          c_col_3 = _mm_fmadd_pd(a_row_k_first_half, b_k3, c_col_3);

          _mm_store_pd(C_ij,         c_col_0);
          _mm_store_pd(C_ij+lda,     c_col_1);
          _mm_store_pd(C_ij+2*lda,   c_col_2);
          _mm_store_pd(C_ij+3*lda,   c_col_3);
        }
      }
      //leftover//
      for (int j = (N/4)*4; j < N; j++){
          c_col_0 = _mm_load_pd(C + i + j*lda);
          __m128d a_row_k_first_half;
          int A_pos = weird_offset_no_multiply(i,0,lda,STRIDE);
          for (int k = 0; k < K; k++){
            a_row_k_first_half  = _mm_load_pd(A + A_pos + k * STRIDE);
            b_k0                = _mm_set1_pd(B[k+j*lda]);
            c_col_0             = _mm_fmadd_pd(a_row_k_first_half, b_k0, c_col_0);
          }
          _mm_store_pd(C+i+j*lda,      c_col_0);
      }
    }
  }

  //Ultimate Leftover, Brute Force
  for (int i = (M/2)*2; i < M; ++i){
    #pragma omp parallel num_threads(1)
    {
      #pragma omp for
      for (int j = 0; j < N; j++){
        c_col_0 = _mm_load_sd(C + i + j*lda);
        __m128d a_row_k_first_half;
        int A_pos = weird_offset_no_multiply(i,0,lda,STRIDE);
        for (int k = 0; k < K; k++){
          a_row_k_first_half  = _mm_load_sd(A + A_pos + k * STRIDE);
          b_k0                = _mm_set1_pd(B[k+j*lda]);
          c_col_0             = _mm_fmadd_sd(a_row_k_first_half, b_k0, c_col_0);
        }
        _mm_store_sd(C+i+j*lda,      c_col_0);
      }
    }
  }
}

// A   M * K
// B   K * N
// C   M * N
static void compute(double* A, double* B, double* C, int M, int N, int K, int lda){
  //The following methods can be called in arbitrary sequence.
  two_by_two_and_naive(A, B, C, M, N, K, lda);
  compute_four_by_four(A, B, C, M, N, K, lda);
  compute_four_by_eight(A, B, C, M, N, K, lda);

}



// A   M * K
// B   K * N
// C   M * N
static void divide_into_small_blocks(double* A, double* B, double* C, int M, int N, int K, int lda){
  
  //tweak size
  int SMALL_M = 8;
  int SMALL_N = 8;
  int SMALL_K = 512;

  //tweak loop
  #pragma omp parallel num_threads(1)
  {
    for (int k = 0; k < K; k += SMALL_K){
      int sub_K = min (SMALL_K, K-k);
      
      #pragma omp parallel num_threads(1)
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
  //printf("Transpose: %f secs\n", x);
  

  //tweak size
  int LARGE_M = 16;
  int LARGE_N = 16;
  int LARGE_K = 1024;

  //tweak loop
  #pragma omp parallel num_threads(min(32, lda / LARGE_M))
  {
    #pragma omp for
    for (int i = 0; i < lda; i += LARGE_M){
      int M = min (LARGE_M, lda-i);

     #pragma omp parallel num_threads(1)
      {
        #pragma omp for
        for (int j = 0; j < lda; j += LARGE_N){
          int N = min (LARGE_N, lda-j);

          #pragma omp parallel num_threads(1)
          {
            #pragma omp for
            for (int k = 0; k < lda; k += LARGE_K){
              int K = min (LARGE_K, lda-k);
              
              divide_into_small_blocks(weird_A + weird_offset(i, k, lda, STRIDE), B+col_major_offset(k, j, lda), C+col_major_offset(i, j, lda), M, N, K, lda);
            }
          }
        }
      }
    }
  }
  free(weird_A);
}

void square_dgemm (int lda, double* A, double* B, double* C){

  double tdata = omp_get_wtime();

  divide_into_large_blocks(A, B, C, lda);

  tdata = omp_get_wtime() - tdata;
  //printf("Total Time: %f secs\n", tdata);

}
