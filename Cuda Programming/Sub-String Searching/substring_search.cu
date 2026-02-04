#include <bits/stdc++.h>
#include <cuda.h>

using namespace std;

#define MAX_STR_LEN 256
#define MAX_QUERY_LEN 128

// =====================================
// LCS Device Function (DP algorithm)
// =====================================
__device__ int lcs_length(char* a, char* b) {
    int n = strlen(a);
    int m = strlen(b);

    int dp[MAX_QUERY_LEN + 1][MAX_STR_LEN + 1];

    for(int i=0;i<=n;i++)
        for(int j=0;j<=m;j++)
            dp[i][j] = 0;

    for(int i=1;i<=n;i++) {
        for(int j=1;j<=m;j++) {
            if(a[i-1] == b[j-1])
                dp[i][j] = dp[i-1][j-1] + 1;
            else
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
        }
    }

    return dp[n][m];
}


// =====================================
// Kernel: LCS Search
// =====================================
__global__ void lcsSearch(char* d_lines,
                          int num_lines,
                          char* search,
                          int threshold) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < num_lines) {

        char* line = d_lines + idx * MAX_STR_LEN;

        int lcs = lcs_length(search, line);

        if(lcs >= threshold) {
            printf("Match (LCS=%d): %s\n", lcs, line);
        }
    }
}


// =====================================
// MAIN
// =====================================
int main(int argc, char* argv[]) {

    if(argc != 4) {
        cout << "Usage: ./prog <search_string> <threads> <threshold>\n";
        return 0;
    }

    string query = argv[1];
    int threads = atoi(argv[2]);
    int threshold = atoi(argv[3]);

    string file_name = "dataset.txt";

    // Read file
    vector<string> lines;
    ifstream file(file_name);

    string line;
    while(getline(file, line)) {
        if(!line.empty())
            lines.push_back(line);
    }

    int n = lines.size();

    // Flatten memory
    char* h_lines = (char*)malloc(n * MAX_STR_LEN);

    for(int i=0;i<n;i++){
        strncpy(h_lines + i*MAX_STR_LEN, lines[i].c_str(), MAX_STR_LEN-1);
        h_lines[i*MAX_STR_LEN + MAX_STR_LEN-1] = '\0';
    }

    // Device memory
    char *d_lines, *d_query;

    cudaMalloc(&d_lines, n * MAX_STR_LEN);
    cudaMalloc(&d_query, query.size()+1);

    cudaMemcpy(d_lines, h_lines, n * MAX_STR_LEN, cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query.c_str(), query.size()+1, cudaMemcpyHostToDevice);

    // Launch
    int blocks = (n + threads - 1) / threads;

    lcsSearch<<<blocks, threads>>>(d_lines, n, d_query, threshold);

    cudaDeviceSynchronize();

    cudaFree(d_lines);
    cudaFree(d_query);
    free(h_lines);

    return 0;
}
