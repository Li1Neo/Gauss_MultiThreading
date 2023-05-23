#ifndef HELPER_UTILS_H_INCLUDED
#define HELPER_UTILS_H_INCLUDED
#include<iostream>
#include<iomanip>
#include<vector>
#include<random>
#define MAXN 2000
using namespace std;

void matrix_print(int n, float m[MAXN][MAXN]) {
    for(int i=0; i<n; ++i) {
        for(int j=0; j<n; ++j) {
            cout<<setw(12)<<m[i][j];
        }
        cout<<endl;
    }
    cout<<endl;
}

void cp_matrix(float m[MAXN][MAXN],float temp_m[MAXN][MAXN]){
    for(int i=0; i<MAXN; ++i) {
        for(int j=0; j<MAXN; ++j) {
            temp_m[i][j]=m[i][j];
        }
    }
}

void random_matrix(int n, float m[MAXN][MAXN]) {
    std::random_device rd;
    for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
            m[i][j] = (float)(rd() % 100);
        }
    }
//    cout<<"随机初始化后的矩阵为:"<<endl;
//    matrix_print(n, m);
}

#endif // HELPER_UTILS_H_INCLUDED
