# **CUDA Programming**

## **1. Introduction**

**CUDA (Compute Unified Device Architecture)** is a parallel computing platform and programming model developed by NVIDIA. It allows developers to leverage the massive parallelism of **Graphics Processing Units (GPUs)** to accelerate computational tasks. Unlike CPUs, which typically have a small number of cores optimized for sequential execution, GPUs have thousands of cores capable of executing thousands of threads concurrently.

**Purpose:** CUDA is used to accelerate data-parallel and compute-intensive tasks, such as matrix multiplication, image processing, scientific simulations, and machine learning.

---

## **2. CUDA Programming Model**

CUDA introduces several **key concepts**:

### **2.1 Host and Device**

* **Host:** The CPU and its memory; runs the main program.
* **Device:** The GPU and its memory; executes parallel kernels.

### **2.2 Kernels**

* A **kernel** is a function executed on the GPU by many threads simultaneously.
* Syntax example in C/C++:

```c
__global__ void myKernel(int *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] *= 2;
}
```

### **2.3 Threads, Blocks, and Grids**

* **Thread:** The smallest execution unit. Each thread executes a copy of the kernel.
* **Block:** A group of threads that can share **fast shared memory**.
* **Grid:** A group of blocks. Provides hierarchical organization for massive parallelism.

**Thread Indexing Example:**

```c
int idx = threadIdx.x + blockIdx.x * blockDim.x;
```

### **2.4 Memory Hierarchy**

1. **Registers:** Private to each thread; fastest access.
2. **Shared Memory:** Shared among threads in the same block; fast.
3. **Global Memory:** Accessible by all threads; large but slower.
4. **Constant and Texture Memory:** Read-only memory for all threads; optimized for certain access patterns.

---

## **3. CUDA Programming Workflow**

1. **Memory Allocation**

   * Allocate memory on GPU using `cudaMalloc`.
2. **Data Transfer**

   * Copy data from **host to device** using `cudaMemcpy`.
3. **Kernel Launch**

   * Launch GPU kernel with a specified **grid and block size**.
4. **Computation**

   * Each thread executes the kernel function on its assigned data.
5. **Result Retrieval**

   * Copy results back to host using `cudaMemcpy`.
6. **Memory Cleanup**

   * Free GPU memory with `cudaFree`.

**Example: Vector Addition**

```c
__global__ void vectorAdd(int *A, int *B, int *C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N) C[idx] = A[idx] + B[idx];
}

int main() {
    int N = 1024;
    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;
    size_t size = N * sizeof(int);

    // Allocate host memory
    h_A = (int*)malloc(size); h_B = (int*)malloc(size); h_C = (int*)malloc(size);

    // Initialize data
    for(int i=0; i<N; i++) { h_A[i] = i; h_B[i] = i; }

    // Allocate device memory
    cudaMalloc(&d_A, size); cudaMalloc(&d_B, size); cudaMalloc(&d_C, size);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
```

---

## **4. Important Concepts**

### **4.1 Thread Indexing**

* Proper thread indexing ensures each thread operates on unique data.

### **4.2 Coalesced Memory Access**

* Arrange global memory accesses so consecutive threads access consecutive memory locations to maximize throughput.

### **4.3 Synchronization**

* `__syncthreads()` is used to synchronize threads within a block when accessing shared memory.

### **4.4 Performance Considerations**

* Maximize occupancy (number of active threads per SM).
* Minimize global memory accesses.
* Use shared memory wisely to reduce latency.

---

## **5. Applications**

* Scientific computing: simulations, PDE solvers
* Linear algebra: matrix multiplication, vector operations
* Image processing: filtering, transformations
* Machine learning: neural network training
* Cryptography and big data analytics

---

## **6. Conclusion**

CUDA allows developers to write parallel programs for GPUs that are **scalable and efficient**. Understanding **thread hierarchy, memory hierarchy, and data parallelism** is essential for writing high-performance CUDA programs.