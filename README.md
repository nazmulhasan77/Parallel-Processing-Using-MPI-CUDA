# **Parallel Processing with CUDA & MPI**

## **1. Introduction**

Parallel processing is a computational technique in which multiple processing elements perform calculations simultaneously, significantly reducing execution time for large-scale problems. This repository demonstrates a hybrid parallel computing approach using:

* **CUDA (Compute Unified Device Architecture):** Enables fine-grained parallelism on GPUs, exploiting thousands of cores to accelerate data-parallel tasks.
* **MPI (Message Passing Interface):** Enables coarse-grained parallelism across multiple CPU nodes in a distributed memory system.

The combination allows efficient utilization of both GPU and CPU resources for high-performance computation.

---

## **2. Theoretical Background**

### **2.1 Parallel Computing Models**

1. **Shared Memory Model (GPU via CUDA)**

   * Threads access a **common memory space** on the GPU.
   * Supports **SIMD (Single Instruction, Multiple Data)** operations: the same instruction executed across multiple data points simultaneously.
   * Memory hierarchy:

     * Registers (fastest)
     * Shared memory (fast)
     * Global memory (slower)

2. **Distributed Memory Model (MPI)**

   * Each process has its **own private memory**.
   * Communication between processes occurs through **message passing**.
   * Ideal for multi-node clusters where data must be exchanged between nodes.

3. **Hybrid Model**

   * Combine CUDA and MPI for **multi-node, multi-GPU computation**.
   * Each node performs GPU computations locally, while MPI coordinates data exchange across nodes.

---

### **2.2 Key Concepts**

* **Task Parallelism:** Different tasks run on different processors simultaneously.

* **Data Parallelism:** The same operation is applied concurrently to multiple data elements.

* **Speedup & Efficiency:**

  $$
  \text{Speedup} = \frac{T_\text{serial}}{T_\text{parallel}}, \quad
  \text{Efficiency} = \frac{\text{Speedup}}{\text{Number of Processors}}
  $$

* **Amdahl’s Law:** Predicts theoretical speedup given the fraction of code that can be parallelized:

  $$
  S_\text{max} = \frac{1}{(1-P) + \frac{P}{N}}
  $$

  where $P$ is the parallelizable portion, $N$ is the number of processors.

* **Memory Coalescing (CUDA):** Organizing memory accesses so consecutive threads access consecutive memory locations for faster GPU execution.

* **Load Balancing (MPI):** Distributing workload evenly across processes to avoid idle processors.

---

## **3. Implementation Overview**

### **3.1 CUDA Implementation**

* **Kernels:** Functions executed on GPU threads.
* **Thread Organization:** Threads grouped into **blocks**, blocks grouped into **grids**.
* **Example:** Matrix multiplication kernel computes each output element by summing products from the corresponding row and column.

### **3.2 MPI Implementation**

* **Process Distribution:** Input data divided among MPI processes.
* **Communication:** Use `MPI_Send` and `MPI_Recv` to exchange boundary or partial results.
* **Synchronization:** Use `MPI_Barrier` to ensure all processes reach the same point before proceeding.

### **3.3 Hybrid Execution**

* Each node runs a CUDA kernel for local computation.
* MPI coordinates results across nodes:

  1. Scatter input data to nodes.
  2. Compute locally on GPU.
  3. Gather results back to root node.

---

## **4. Example Applications**

1. **Matrix Multiplication:** Compute $C = A \times B$ using parallel threads.
2. **n-Body Simulation:** Update positions of particles in parallel to simulate gravitational forces.
3. **Mandelbrot Set Computation:** Evaluate pixels independently for fractal generation.
4. **Word Count / Pattern Matching:** Split text into segments and process in parallel.

---

## **5. Performance Analysis**

* **Metrics:** Execution time, speedup, efficiency.
* **Strong Scaling:** How execution time changes with more processors for a fixed problem size.
* **Weak Scaling:** How execution time changes as problem size and processors increase proportionally.

---

## **6. Conclusion**

By combining **CUDA** for GPU acceleration and **MPI** for distributed memory computation, this hybrid approach demonstrates:

* Efficient utilization of computational resources.
* Scalability to large datasets and clusters.
* Practical application of parallel computing theories in real-world scenarios.