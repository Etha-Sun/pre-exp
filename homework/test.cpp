#include <iostream>
#include <algorithm> 
using namespace std;

int main() {
    long long L;
    cin >> L;

    if (L == 1) {
        cout << 0 << endl;
        return 0;
    }

    long long x = L - 1;
    int k = 0;
    while ((1LL << (k + 1)) <= x) {
        k++;
    }
    int m = k + 1;
    long long s = 1LL << m; 
    long long T = m * L - (s - 1);

    long long S = L * (L - 1) / 2;

    cout << min(T, S) << endl;

    return 0;
}