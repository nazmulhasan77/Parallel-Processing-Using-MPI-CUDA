// File: mpi_cuda_lcs.cpp


// # Compile
// mpicxx -o mpi_cuda_lcs mpi_cuda_lcs.cpp 
// # Run with 4 MPI processes
// mpirun -np 4 ./mpi_cuda_lcs



#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>

#define MAX_LINE_LENGTH 1024

// CUDA kernel to compute LCS length of each line with input string
__global__ void lcs_kernel(char *lines, int *line_offsets, char *input, int input_len, int *results, int num_lines) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_lines) return;

    int start = line_offsets[idx];
    int end = line_offsets[idx + 1];
    int line_len = end - start;

    extern __shared__ int dp[];
    int *prev = dp;
    int *curr = dp + input_len;

    int max_len = 0;

    for (int i = 0; i <= input_len; i++) prev[i] = 0;

    for (int i = 0; i < line_len; i++) {
        curr[0] = 0;
        for (int j = 0; j < input_len; j++) {
            if (lines[start + i] == input[j])
                curr[j + 1] = prev[j] + 1;
            else
                curr[j + 1] = 0;
            if (curr[j + 1] > max_len) max_len = curr[j + 1];
        }
        // Swap prev and curr
        int *tmp = prev;
        prev = curr;
        curr = tmp;
    }

    results[idx] = max_len;
}

// Utility: Read all lines from a file
std::vector<std::string> read_lines(const std::string &filename) {
    std::ifstream infile(filename);
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(infile, line)) {
        if (!line.empty()) lines.push_back(line);
    }
    return lines;
}

// Flatten lines into a single char array and compute offsets
void flatten_lines(const std::vector<std::string> &lines, std::vector<char> &flat_lines, std::vector<int> &offsets) {
    offsets.push_back(0);
    for (auto &line : lines) {
        flat_lines.insert(flat_lines.end(), line.begin(), line.end());
        offsets.push_back(flat_lines.size());
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<std::string> all_lines;
    std::string input_str;

    if (rank == 0) {
        // Master: Read file and user input
        all_lines = read_lines("dataset.txt");
        std::cout << "Enter input string: ";
        std::cin >> input_str;
    }

    // Broadcast input string length and string
    int input_len = input_str.size();
    MPI_Bcast(&input_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) input_str.resize(input_len);
    MPI_Bcast(&input_str[0], input_len, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Scatter lines across processes
    int total_lines = all_lines.size();
    MPI_Bcast(&total_lines, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int chunk_size = (total_lines + size - 1) / size; // ceiling division
    int start_idx = rank * chunk_size;
    int end_idx = std::min(start_idx + chunk_size, total_lines);

    std::vector<std::string> local_lines;
    if (rank == 0) {
        for (int r = 1; r < size; r++) {
            int r_start = r * chunk_size;
            int r_end = std::min(r_start + chunk_size, total_lines);
            int num_lines_r = r_end - r_start;
            // Send number of lines
            MPI_Send(&num_lines_r, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
            for (int i = r_start; i < r_end; i++) {
                int len = all_lines[i].size();
                MPI_Send(&len, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
                MPI_Send(all_lines[i].c_str(), len, MPI_CHAR, r, 0, MPI_COMM_WORLD);
            }
        }
        // Master keeps its own chunk
        local_lines.insert(local_lines.end(), all_lines.begin() + start_idx, all_lines.begin() + end_idx);
    } else {
        int num_local_lines;
        MPI_Recv(&num_local_lines, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < num_local_lines; i++) {
            int len;
            MPI_Recv(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            char buffer[MAX_LINE_LENGTH];
            MPI_Recv(buffer, len, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            buffer[len] = '\0';
            local_lines.push_back(std::string(buffer));
        }
    }

    // Flatten local lines for GPU
    std::vector<char> flat_lines;
    std::vector<int> line_offsets;
    flatten_lines(local_lines, flat_lines, line_offsets);
    int num_local_lines = local_lines.size();

    // Allocate GPU memory
    char *d_lines, *d_input;
    int *d_offsets, *d_results;
    cudaMalloc(&d_lines, flat_lines.size() * sizeof(char));
    cudaMalloc(&d_input, input_len * sizeof(char));
    cudaMalloc(&d_offsets, line_offsets.size() * sizeof(int));
    cudaMalloc(&d_results, num_local_lines * sizeof(int));

    cudaMemcpy(d_lines, flat_lines.data(), flat_lines.size() * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, input_str.data(), input_len * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, line_offsets.data(), line_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    int threads = 256;
    int blocks = (num_local_lines + threads - 1) / threads;
    size_t shared_mem = 2 * (input_len + 1) * sizeof(int); // for prev and curr rows
    lcs_kernel<<<blocks, threads, shared_mem>>>(d_lines, d_offsets, d_input, input_len, d_results, num_local_lines);
    cudaDeviceSynchronize();

    // Copy results back
    std::vector<int> lcs_lengths(num_local_lines);
    cudaMemcpy(lcs_lengths.data(), d_results, num_local_lines * sizeof(int), cudaMemcpyDeviceToHost);

    // Collect matching lines
    std::vector<std::string> matched_lines;
    for (int i = 0; i < num_local_lines; i++) {
        if (lcs_lengths[i] > 0) matched_lines.push_back(local_lines[i]);
    }

    // Gather results at master
    if (rank == 0) {
        std::vector<std::string> final_lines = matched_lines;
        for (int r = 1; r < size; r++) {
            int num_recv;
            MPI_Recv(&num_recv, 1, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < num_recv; i++) {
                int len;
                MPI_Recv(&len, 1, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                char buffer[MAX_LINE_LENGTH];
                MPI_Recv(buffer, len, MPI_CHAR, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                buffer[len] = '\0';
                final_lines.push_back(std::string(buffer));
            }
        }
        // Write output
        std::ofstream outfile("output.txt");
        for (auto &line : final_lines) outfile << line << "\n";
        std::cout << "Matched lines written to output.txt\n";
    } else {
        int num_send = matched_lines.size();
        MPI_Send(&num_send, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        for (auto &line : matched_lines) {
            int len = line.size();
            MPI_Send(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(line.c_str(), len, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    }

    cudaFree(d_lines);
    cudaFree(d_input);
    cudaFree(d_offsets);
    cudaFree(d_results);

    MPI_Finalize();
    return 0;
}
