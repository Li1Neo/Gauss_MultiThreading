#include<immintrin.h>
#include "helper_utils.h"
#define MAXN 2000
using namespace std;


//未对齐的AVX的高斯消去法
void Gauss_AVX_unaligned(int n, float m[MAXN][MAXN]) {
  __m256 mmt1, mmt2, mmt3, mmt4; // 4个单精度浮点数的向量寄存器
  for (int i = 0; i<n; ++i) {
      //处理第j行
      float t1 = m[i][i];
      mmt1 = _mm256_set1_ps(t1);
      int j = i;
      for(; j<n; j+=8) { // 8个一组进行计算
          mmt2 = _mm256_loadu_ps(&m[i][j]);
          mmt3 = _mm256_div_ps(mmt2, mmt1);
          _mm256_storeu_ps(&m[i][j], mmt3);
      }
      for(; j<n; ++j) { // 处理不能并行化的剩余部分
          m[i][j] = m[i][j] / t1;
      }
      //处理j+1~n行
      for(int j=i+1; j<n; ++j) {
          float t2 = m[j][i];
          mmt1 = _mm256_set1_ps(t2);
          int k = i+1;
          for (; k<n; k+=8) {
              mmt2 = _mm256_loadu_ps(&m[j][k]);
              mmt3 = _mm256_loadu_ps(&m[i][k]);
              mmt4 = _mm256_sub_ps(mmt2, _mm256_mul_ps(mmt1, mmt3));
              _mm256_storeu_ps(&m[j][k], mmt4);
          }
          for (; k<n; ++k) { // 处理不能并行化的剩余部分
              m[j][k] = m[j][k] - t2 * m[i][k];
          }
          m[j][i] = 0.0;
      }
//      matrix_print(n, m);
  }
}

//未对齐的AVX的高斯消去法(只对第一个内嵌循环优化)
void Gauss_AVX_unaligned1(int n, float m[MAXN][MAXN]) {
  __m256 mmt1, mmt2, mmt3, mmt4;
  for (int i = 0; i<n; ++i) {
      //处理第j行
      float t1 = m[i][i];
      mmt1 = _mm256_set1_ps(t1);
      int j = i;
      for(; j<n; j+=8) { // 8个一组进行计算
          mmt2 = _mm256_loadu_ps(&m[i][j]);
          mmt3 = _mm256_div_ps(mmt2, mmt1);
          _mm256_storeu_ps(&m[i][j], mmt3);
      }
      for(; j<n; ++j) { // 处理不能并行化的剩余部分
          m[i][j] = m[i][j] / t1;
      }
      //处理j+1~n行
      for (int j=i+1; j<n; ++j) { //对第j+1~n行消元
            float t2 = m[j][i];
            for (int k=i+1; k<n; ++k) {
                m[j][k] = m[j][k] - t2 * m[i][k];
            }
            m[j][i] = 0.0;
      }
//      matrix_print(n, m);
  }
}

//未对齐的AVX的高斯消去法(只对第二个内嵌循环优化)
void Gauss_AVX_unaligned2(int n, float m[MAXN][MAXN]) {
  __m256 mmt1, mmt2, mmt3, mmt4;
  for (int i = 0; i<n; ++i) {
      float t1 = m[i][i];
      for (int j=i; j<n; ++j) { //第j行让第一个非零元素变为1
            m[i][j] = m[i][j] / t1;
    }
      //处理j+1~n行
      for(int j=i+1; j<n; ++j) {
          float t2 = m[j][i];
          mmt1 = _mm256_set1_ps(t2);
          int k = i+1;
          for (; k<n; k+=8) {
              mmt2 = _mm256_loadu_ps(&m[j][k]);
              mmt3 = _mm256_loadu_ps(&m[i][k]);
              mmt4 = _mm256_sub_ps(mmt2, _mm256_mul_ps(mmt1, mmt3));
              _mm256_storeu_ps(&m[j][k], mmt4);
          }
          for (; k<n; ++k) { // 处理不能并行化的剩余部分
              m[j][k] = m[j][k] - t2 * m[i][k];
          }
          m[j][i] = 0.0;
      }
//      matrix_print(n, m);
  }
}
