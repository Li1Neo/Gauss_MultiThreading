#include<iostream>
#include<iomanip>
#include<stdio.h>
#include<cmath>
#include"helper_utils.h"
#define N 2000
using namespace std;

void Gauss_serial(int n, float m[N][N]) {
    for (int i=0; i<n; ++i) {
        float t1 = m[i][i];
        for (int j=i; j<n; ++j) { //第j行让第一个非零元素变为1
            m[i][j] = m[i][j] / t1;
        }
        for (int j=i+1; j<n; ++j) { //对第j+1~n行消元
            float t2 = m[j][i];
            for (int k=i+1; k<n; ++k) {
                m[j][k] = m[j][k] - t2 * m[i][k];
            }
            m[j][i] = 0.0;
        }
//        matrix_print(n, m);
    }
}
