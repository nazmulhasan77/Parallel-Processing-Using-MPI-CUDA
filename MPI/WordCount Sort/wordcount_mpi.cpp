
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cctype>

using namespace std;

// Convert a string to lowercase and remove punctuation
string normalize(const string &word) {
    string result;
    for (char c : word) {
        if (isalnum(c)) result += tolower(c);
    }
    return result;
}

// Merge two unordered_maps by adding frequencies
void mergeMaps(unordered_map<string, int> &globalMap, const unordered_map<string, int> &localMap) {
    for (auto &p : localMap) {
        globalMap[p.first] += p.second;
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) cout << "Usage: mpirun -np <num_processes> " << argv[0] << " <filename>" << endl;
        MPI_Finalize();
        return 1;
    }

    double start_time = MPI_Wtime();

    string filename = argv[1];
    vector<string> lines;

    // Only rank 0 reads the file
    if (rank == 0) {
        ifstream file(filename);
        if (!file.is_open()) {
            cout << "Error opening file: " << filename << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        string line;
        while (getline(file, line)) {
            lines.push_back(line);
        }
        file.close();
    }

    // Broadcast total number of lines to all processes
    int total_lines = lines.size();
    MPI_Bcast(&total_lines, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute chunk size for each process
    int chunk_size = (total_lines + size - 1) / size;

    // Send chunks to each process
    vector<string> local_lines(chunk_size);
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            int start = i * chunk_size;
            int end = min(start + chunk_size, total_lines);
            vector<string> temp_lines;
            for (int j = start; j < end; j++) temp_lines.push_back(lines[j]);
            if (i == 0) {
                local_lines = temp_lines;
            } else {
                // Send number of lines first
                int n = temp_lines.size();
                MPI_Send(&n, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                // Send actual lines
                for (const auto &l : temp_lines) {
                    int len = l.size();
                    MPI_Send(&len, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                    MPI_Send(l.c_str(), len, MPI_CHAR, i, 0, MPI_COMM_WORLD);
                }
            }
        }
    } else {
        int n;
        MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_lines.resize(n);
        for (int i = 0; i < n; i++) {
            int len;
            MPI_Recv(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            char *buffer = new char[len + 1];
            MPI_Recv(buffer, len, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            buffer[len] = '\0';
            local_lines[i] = string(buffer);
            delete[] buffer;
        }
    }

    // Each process counts words locally
    unordered_map<string, int> local_map;
    for (auto &line : local_lines) {
        stringstream ss(line);
        string word;
        while (ss >> word) {
            word = normalize(word);
            if (!word.empty()) local_map[word]++;
        }
    }

    // Reduce all maps to rank 0
    unordered_map<string, int> global_map;
    if (rank == 0) global_map = local_map;

    // Serialize maps to send
    if (rank != 0) {
        int map_size = local_map.size();
        MPI_Send(&map_size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        for (auto &p : local_map) {
            int len = p.first.size();
            MPI_Send(&len, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(p.first.c_str(), len, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
            MPI_Send(&p.second, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        }
    } else {
        for (int i = 1; i < size; i++) {
            int map_size;
            MPI_Recv(&map_size, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < map_size; j++) {
                int len;
                MPI_Recv(&len, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                char *buffer = new char[len + 1];
                MPI_Recv(buffer, len, MPI_CHAR, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                buffer[len] = '\0';
                string key(buffer);
                delete[] buffer;
                int value;
                MPI_Recv(&value, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                global_map[key] += value;
            }
        }
    }

    double end_time = MPI_Wtime();

    // Rank 0 prints top 10 words and time
    if (rank == 0) {
        vector<pair<string, int>> vec(global_map.begin(), global_map.end());
        sort(vec.begin(), vec.end(), [](auto &a, auto &b) { return a.second > b.second; });

        cout << "Total processing time: " << end_time - start_time << " seconds\n";
        cout << "Top 10 word occurrences:\n";
        for (int i = 0; i < min(10, (int)vec.size()); i++) {
            cout << vec[i].first << " : " << vec[i].second << endl;
        }
    }

    MPI_Finalize();
    return 0;
}
//
// mpic++ -o wordcount_mpi wordcount_mpi.cpp
//mpirun -np 4 ./wordcount_mpi input.txt
