#include<immintrin.h>
#include <omp.h>
#include "helper_utils.h"
#define MAXN 2000
#define thread_num 8
#define chunk 100
using namespace std;

//AVX高斯消去法—-OpenMP优化
void Gauss_AVX_unaligned_omp_guided(int n, float m[MAXN][MAXN]) {
    __m256 mmt1, mmt2, mmt3, mmt4; //4个单精度浮点数的向量寄存器
    #pragma omp parallel num_threads(thread_num)
    for (int i=0; i<n; ++i) {
        float t1 = m[i][i];
        //处理第i行
        #pragma omp schedule(guided)
        for (int j=i+1; j<n; ++j) { //第i行让第一个非零元素变为1
            m[i][j] = m[i][j] / t1;
        }
        // j 不能从i开始，因为可能导致一个线程修改了m[i][i], 而另一个线程又将m[i][i]当成除数，造成线程竞争
        m[i][i] = 1;
        //处理i+1~n行
        #pragma omp for schedule(guided) private(mmt1, mmt2, mmt3, mmt4)
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
    }
}


//高斯消去法 —-OpenMP优化(static)
void Gauss_omp_static(int n, float m[MAXN][MAXN]) {
    #pragma omp parallel num_threads(thread_num)
    for (int i=0; i<n; ++i) {
        float t1 = m[i][i];
        #pragma omp for schedule(static, chunk)
        for (int j=i+1; j<n; ++j) { //第j行让第一个非零元素变为1
            m[i][j] = m[i][j] / t1;
        }
        m[i][i] = 1;
        #pragma omp for schedule(static, chunk)
        for (int j=i+1; j<n; ++j) { //对第j+1~n行消元
            float t2 = m[j][i];
            for (int k=i+1; k<n; ++k) {
                m[j][k] = m[j][k] - t2 * m[i][k];
            }
            m[j][i] = 0.0;
        }
    }
}


//高斯消去法 —-OpenMP优化(dynamic)
void Gauss_omp_dynamic(int n, float m[MAXN][MAXN]) {
    #pragma omp parallel num_threads(thread_num)
    for (int i=0; i<n; ++i) {
        float t1 = m[i][i];
        #pragma omp for schedule(dynamic, chunk)
        for (int j=i+1; j<n; ++j) { //第j行让第一个非零元素变为1
            m[i][j] = m[i][j] / t1;
        }
        m[i][i] = 1;
        #pragma omp for schedule(dynamic, chunk)
        for (int j=i+1; j<n; ++j) { //对第j+1~n行消元
            float t2 = m[j][i];
            for (int k=i+1; k<n; ++k) {
                m[j][k] = m[j][k] - t2 * m[i][k];
            }
            m[j][i] = 0.0;
        }
    }
}


//高斯消去法 —-OpenMP优化(guided)
void Gauss_omp_guided(int n, float m[MAXN][MAXN]) {
    #pragma omp parallel num_threads(thread_num)
    for (int i=0; i<n; ++i) {
        float t1 = m[i][i];
        #pragma omp for schedule(guided)
        for (int j=i+1; j<n; ++j) { //第j行让第一个非零元素变为1
            m[i][j] = m[i][j] / t1;
        }
        m[i][i] = 1;
        #pragma omp for schedule(guided)
        for (int j=i+1; j<n; ++j) { //对第j+1~n行消元
            float t2 = m[j][i];
            for (int k=i+1; k<n; ++k) {
                m[j][k] = m[j][k] - t2 * m[i][k];
            }
            m[j][i] = 0.0;
        }
    }
}
