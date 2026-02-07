## **Basic Structure of an MPI Program**

### 1. **Include MPI Header**

```c
#include <mpi.h>
```

This includes all necessary MPI functions and constants.

---

### 2. **Initialize MPI**

```c
MPI_Init(&argc, &argv);
```

* **Purpose:** Starts the MPI environment.
* **Arguments:** `argc` and `argv` from `main()`, so MPI can process command-line arguments if needed.

---

### 3. **Get Rank and Size**

```c
int world_rank, world_size;

MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Get the process ID
MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Get total number of processes
```

* **MPI_Comm_rank:** Returns the unique rank (ID) of the process (0,1,2,...,size-1).
* **MPI_Comm_size:** Returns the total number of processes in the communicator (`MPI_COMM_WORLD` is the default global communicator).

---

### 4. **Communication Between Processes**

MPI provides **point-to-point** communication and **collective communication**.

**Example: Point-to-Point Communication**

```c
int number;
if (world_rank == 0) {
    number = 42;
    MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
} else if (world_rank == 1) {
    MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Process 1 received number %d from process 0\n", number);
}
```

**Explanation of MPI_Send/MPI_Recv arguments:**

1. `&number` → pointer to the data to send/receive.
2. `1` → number of elements.
3. `MPI_INT` → data type (can be `MPI_FLOAT`, `MPI_DOUBLE`, `MPI_CHAR`, etc.).
4. Destination/Source rank (`0`, `1`, ...).
5. `tag` → integer to label messages (used to differentiate message types, e.g., `0`).
6. `MPI_COMM_WORLD` → communicator.
7. `MPI_STATUS_IGNORE` → ignores the status output (can be used to get info about the message).

---

### 5. **Finalize MPI**

```c
MPI_Finalize();
```

* **Purpose:** Ends the MPI environment. No MPI function should be called after this.

---

## **Complete Minimal MPI Example**

```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    // 1. Initialize MPI
    MPI_Init(&argc, &argv);

    // 2. Get rank and size
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    printf("Hello from process %d of %d\n", world_rank, world_size);

    // 3. Simple communication: send number from 0 to 1
    if (world_rank == 0) {
        int number = 42;
        MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("Process 0 sent number %d to process 1\n", number);
    } else if (world_rank == 1) {
        int number;
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 1 received number %d from process 0\n", number);
    }

    // 4. Finalize MPI
    MPI_Finalize();
    return 0;
}
```

---

### **How to Compile and Run**

```bash
mpicc -o mpi_example mpi_example.c   # Compile
mpirun -np 2 ./mpi_example           # Run with 2 processes
```

**Expected Output:**

```
Hello from process 0 of 2
Hello from process 1 of 2
Process 0 sent number 42 to process 1
Process 1 received number 42 from process 0
```

---

✅ **Summary of Important MPI Functions**

| Function                     | Purpose                              |
| ---------------------------- | ------------------------------------ |
| `MPI_Init(&argc, &argv)`     | Initialize MPI                       |
| `MPI_Comm_size(comm, &size)` | Get number of processes              |
| `MPI_Comm_rank(comm, &rank)` | Get rank (ID) of process             |
| `MPI_Send(...)`              | Send message to another process      |
| `MPI_Recv(...)`              | Receive message from another process |
| `MPI_Finalize()`             | Finalize MPI                         |

---
