# 🧩 MPI Theory

## 🔹 1. What is MPI?

* **MPI (Message Passing Interface)** is a **standard** (not a library itself) that defines a set of functions for communication between processes in **parallel and distributed computing**.
* Implementations: **Open MPI**, **MPICH**, Intel MPI, MVAPICH, etc.
* Works in **distributed memory systems** (clusters, supercomputers) where processes don’t share memory but exchange information via **messages**.

---

## 🔹 2. Parallel Programming Models

* **Shared Memory** (OpenMP, Pthreads): All processes/threads see the same memory space.
* **Distributed Memory** (MPI): Each process has its own memory; communication is explicit through send/receive.
* **Hybrid Model**: MPI + OpenMP/CUDA (common in HPC).

MPI belongs to **Distributed Memory Model**.

---

## 🔹 3. MPI Process Model

* **SPMD (Single Program Multiple Data)** is the most common model.

  * Each process runs the **same program**, but operates on **different data**.
  * Processes are distinguished by their **rank**.

Example:

* Process 0 → read input, distribute data.
* Process 1, 2, 3… → perform computations in parallel.

---

## 🔹 4. MPI Basics

### a) Initialization & Finalization

Every MPI program must:

1. **Start MPI environment**: `MPI_Init()`
2. **Determine process rank & size**:

   * `MPI_Comm_rank` → unique ID of process (0…N-1).
   * `MPI_Comm_size` → total number of processes.
3. **End environment**: `MPI_Finalize()`

---

### b) Communication Types

1. **Point-to-Point Communication**

   * One process sends a message, another receives it.
   * Functions: `MPI_Send`, `MPI_Recv`
   * Can be **blocking** (waits until complete) or **nonblocking** (`MPI_Isend`, `MPI_Irecv`).

2. **Collective Communication**

   * Involves all processes in a communicator.
   * Examples:

     * `MPI_Bcast` → broadcast from one process to all.
     * `MPI_Scatter` → split data from root to processes.
     * `MPI_Gather` → collect data from processes.
     * `MPI_Reduce` → combine values (sum, min, max, etc.).

3. **Synchronization**

   * `MPI_Barrier` → all processes wait until everyone reaches this point.

---

### c) Communicators

* A **communicator** defines a group of processes that can talk to each other.
* Default communicator: `MPI_COMM_WORLD` (all processes).
* Can create sub-communicators for specific tasks.

---

### d) Derived Data Types

* MPI allows custom **data types** for sending structured data (arrays, structs).
* Functions: `MPI_Type_contiguous`, `MPI_Type_vector`, `MPI_Type_create_struct`.

---

## 🔹 5. MPI Execution

* MPI programs are usually run with:

  ```bash
   mpicc Matrix_Multiplication.c -o Matrix_Multiplication
  ```
  → Compile `program`.
  ```bash
   mpiexec -n 4 ./Matrix_Multiplication
  ```

  → Runs `program` with 4 processes.
* Each process has its **own memory space** and communicates using MPI calls.

---

## 🔹 6. Advantages of MPI

* Portable: Runs on any distributed system.
* Scalable: Works from a few processes to hundreds of thousands.
* Flexibility: Fine control over communication.

---

## 🔹 7. Limitations of MPI

* **Programming complexity**: Explicit message passing is harder than shared memory.
* **Communication cost**: Data exchange over a network is slower than shared memory.
* **Debugging difficulty**: Many processes = harder to track errors.

---

## 🔹 8. MPI vs Other Models

| Feature       | Shared Memory (OpenMP) | Distributed Memory (MPI) |
| ------------- | ---------------------- | ------------------------ |
| Memory access | Global shared memory   | Private per process      |
| Communication | Implicit (via memory)  | Explicit (messages)      |
| Scalability   | Limited (one machine)  | High (across clusters)   |

---

✅ **In short:**
MPI is the backbone of **High Performance Computing (HPC)**. It provides a standard way for processes in **distributed systems** to communicate and coordinate via message passing, enabling massive parallelism in scientific simulations, weather forecasting, machine learning, etc.
