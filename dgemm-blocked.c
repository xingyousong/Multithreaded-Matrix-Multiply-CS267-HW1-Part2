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
 
const char* dgemm_desc = "Yao & Xingyou's OpenMP WIP blocked dgemm.";

double* transpose(double* src, int lda){
  __assume_aligned(src, 32);
  double* dest __attribute((aligned(32))) = malloc(lda * lda * sizeof(double));
  for (int i = 0; i < lda; i++){
    for (int j = 0; j < lda; j++){
      dest[i * lda + j] = src[j *  lda + i];
    }
  }
  return dest;
}

void SSE_dot(double* restrict A, double* restrict B, double* restrict C, int l){
  int i = 0;
  
  for (;i < l - 4; i+= 4){
    __m256d xy0 = _mm256_mul_pd( x[i], y[i] );
    __m256d xy1 = _mm256_mul_pd( x[i+1], y[i+1] );
    __m256d xy2 = _mm256_mul_pd( x[i+2], y[i+2] );
    __m256d xy3 = _mm256_mul_pd( x[i+3], y[i+3] );

    // low to high: xy00+xy01 xy10+xy11 xy02+xy03 xy12+xy13
    __m256d temp01 = _mm256_hadd_pd( xy0, xy1 );   

    // low to high: xy20+xy21 xy30+xy31 xy22+xy23 xy32+xy33
    __m256d temp23 = _mm256_hadd_pd( xy2, xy3 );

    // low to high: xy02+xy03 xy12+xy13 xy20+xy21 xy30+xy31
    __m256d swapped = _mm256_permute2f128_pd( temp01, temp23, 0x21 );

    // low to high: xy00+xy01 xy10+xy11 xy22+xy23 xy32+xy33
    __m256d blended = _mm256_blend_pd(temp01, temp23, 0b1100);

    __m256d dotproduct = _mm256_add_pd( swapped, blended );
  }
  for (i < l; i++){
    val += A[i] * B[i];
  }
  C[0] = val;
  
}

void dot_product(double* restrict A, double* restrict B, double* restrict C, int l){
  __assume_aligned(A, 32);
  __assume_aligned(B, 32);
  __assume_aligned(C, 32);
  double val = C[0];
  

}


void square_dgemm (int lda, double* A, double* B, double* C){
  __assume_aligned(A, 32);
  __assume_aligned(B, 32);
  __assume_aligned(C, 32);
  double* A_transpose __attribute((aligned(32))) = transpose(A, lda);
  
  
  #pragma omp parallel num_threads(16)
  {
    #pragma omp for
    for (int j = 0; j < lda; j++){
      B[j * lda] = B[j * lda];
      A[0] = A[0];

      #pragma omp parallel num_threads(1)
      {
        #pragma omp for
        for (int i = 0; i < lda; i++){
          C[j * lda] = C[j * lda];
          dot_product(A_transpose + i * lda, B + j * lda, C + j * lda + i, lda);
        }
      }
    }
  }
  free(A_transpose);
}