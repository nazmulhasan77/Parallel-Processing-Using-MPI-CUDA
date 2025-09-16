/*
    Problem: Find the longest common substring between a search term and names in a phonebook, IGNORING CASE.
    Implementation: Parallelized using MPI.

    How to run:
    1. Compile: mpic++ -o mpi_phonebook_lcs mpi_phonebook_lcs.cpp
    2. Run:     mpirun -np 4 ./mpi_phonebook_lcs phonebook.txt "sumaiya"

    Example:
    mpirun -np 4 ./mpi_phonebook_lcs phonebook.txt "sumaiya akter"
*/

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>

using namespace std;

// Convert string to lowercase
string toLower(const string &s) {
    string result = s;
    transform(result.begin(), result.end(), result.begin(),
              [](unsigned char c){ return tolower(c); });
    return result;
}

// Compute Longest Common Substring length
int lcsLength(const string &a, const string &b) {
    int n = a.size(), m = b.size();
    vector<vector<int>> dp(n+1, vector<int>(m+1, 0));
    int maxLen = 0;

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            if (a[i-1] == b[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
                maxLen = max(maxLen, dp[i][j]);
            }
        }
    }
    return maxLen;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 3) {
        if (world_rank == 0) {
            cerr << "Usage: mpirun -np <num_processes> "
                 << argv[0] << " <phonebook_file> <search_term>" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    string filename = argv[1];
    string searchTerm = toLower(argv[2]);
    vector<string> phonebook;

    // Rank 0 reads phonebook file
    if (world_rank == 0) {
        ifstream file(filename);
        if (!file) {
            cerr << "Error: Cannot open file " << filename << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        string line;
        while (getline(file, line)) {
            if (!line.empty()) {
                phonebook.push_back(line);
            }
        }
        file.close();
    }

    // Broadcast phonebook size
    int phonebookSize = phonebook.size();
    MPI_Bcast(&phonebookSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter work among processes
    int chunkSize = (phonebookSize + world_size - 1) / world_size;
    vector<string> local_entries;

    // Pack data for scattering
    if (world_rank == 0) {
        // Ensure all processes get approx same size chunks
        for (int i = 0; i < world_size; i++) {
            int start = i * chunkSize;
            int end = min(start + chunkSize, phonebookSize);

            if (i == 0) {
                local_entries.insert(local_entries.end(),
                                     phonebook.begin() + start,
                                     phonebook.begin() + end);
            } else {
                vector<string> temp(phonebook.begin() + start,
                                    phonebook.begin() + end);

                // Serialize and send
                int count = temp.size();
                MPI_Send(&count, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                for (auto &s : temp) {
                    int len = s.size();
                    MPI_Send(&len, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                    MPI_Send(s.c_str(), len, MPI_CHAR, i, 0, MPI_COMM_WORLD);
                }
            }
        }
    } else {
        int count;
        MPI_Recv(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < count; i++) {
            int len;
            MPI_Recv(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            string s(len, ' ');
            MPI_Recv(&s[0], len, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            local_entries.push_back(s);
        }
    }

    // Local computation
    int local_best_len = 0;
    string local_best_name;
    for (auto &entry : local_entries) {
        int len = lcsLength(toLower(entry), searchTerm);
        if (len > local_best_len) {
            local_best_len = len;
            local_best_name = entry;
        }
    }

    // Gather results at root
    struct {
        int length;
        int rank;
    } local_result, global_result;

    local_result.length = local_best_len;
    local_result.rank = world_rank;

    MPI_Allreduce(&local_result, &global_result, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

    // Root process prints final result
    if (world_rank == global_result.rank) {
        cout << "Best match: " << local_best_name
             << " (LCS length = " << local_best_len << ")" << endl;
    }

    MPI_Finalize();
    return 0;
}
