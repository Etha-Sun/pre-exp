#include <iostream>
#include <algorithm> // 用于min函数
using namespace std;

int main() {
    long long L; // 用long long避免大数值溢出
    cin >> L;

    if (L == 1) {
        cout << 0 << endl;
        return 0;
    }

    long long x = L - 1;
    int k = 0;
    // 找到最大的k使得2^(k+1) <= x（使用1LL确保移位操作不溢出）
    while ((1LL << (k + 1)) <= x) {
        k++;
    }
    int m = k + 1;
    long long s = 1LL << m; // 2^m
    long long T = m * L - (s - 1);

    // 计算选择、插入、冒泡排序的最坏比较次数S
    long long S = L * (L - 1) / 2;

    // 输出最小值
    cout << min(T, S) << endl;

    return 0;
}