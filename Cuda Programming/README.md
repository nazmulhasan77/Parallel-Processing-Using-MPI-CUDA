## **Basic Structure of a CUDA Program**

### 1. **Include CUDA Header**

```c
#include <cuda_runtime.h>
#include <stdio.h>
```

* `cuda_runtime.h` provides all the CUDA runtime APIs.
* `stdio.h` is for printing output on CPU (host).

---

### 2. **Define a Kernel**

A **kernel** is a function that runs on the GPU.

```c
__global__ void addKernel(int *c, const int *a, const int *b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

**Explanation of the kernel:**

* `__global__` → specifies this function runs on **GPU** and is called from **CPU**.
* `threadIdx.x` → thread index **within a block**.
* `blockIdx.x` → block index **within the grid**.
* `blockDim.x` → number of threads per block.
* The combination `threadIdx.x + blockIdx.x * blockDim.x` gives the **global thread index**.
* `if (i < n)` → prevents out-of-bound memory access.

---

### 3. **Main Function (Host Code)**

```c
int main() {
    int n = 10;
    size_t size = n * sizeof(int);

    // 1. Allocate host memory
    int h_a[n], h_b[n], h_c[n];
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // 2. Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // 3. Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 4. Launch kernel
    int threadsPerBlock = 4;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_c, d_a, d_b, n);

    // 5. Copy result from device to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // 6. Print result
    for (int i = 0; i < n; i++) {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }

    // 7. Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
```

---

### **Step-by-Step Explanation**

| Step | CUDA Function / Concept  | Explanation                                                      |
| ---- | ------------------------ | ---------------------------------------------------------------- |
| 1    | Host memory allocation   | `int h_a[n];` allocated on CPU                                   |
| 2    | Device memory allocation | `cudaMalloc(&d_a, size);` allocates GPU memory                   |
| 3    | Copy data to GPU         | `cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);`            |
| 4    | Launch kernel            | `kernel<<<blocksPerGrid, threadsPerBlock>>>(...);` runs GPU code |
| 5    | Copy result back to CPU  | `cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);`            |
| 6    | Print result             | Regular CPU code to check output                                 |
| 7    | Free GPU memory          | `cudaFree(d_a);` avoids memory leaks                             |

---

### **Kernel Launch Syntax**

```c
kernel<<<numBlocks, threadsPerBlock>>>(arguments);
```

* `numBlocks` → number of blocks in the grid.
* `threadsPerBlock` → number of threads per block.
* Total threads = `numBlocks * threadsPerBlock`.

---

### **Compile and Run**

```bash
nvcc -o cuda_example cuda_example.cu   # Compile CUDA code
./cuda_example                          # Run program
```

**Expected Output:**

```
0 + 0 = 0
1 + 2 = 3
2 + 4 = 6
3 + 6 = 9
...
```

---

### ✅ **Key CUDA Concepts**

| Concept      | Description                            |
| ------------ | -------------------------------------- |
| Host         | CPU memory and execution               |
| Device       | GPU memory and execution               |
| Kernel       | GPU function, runs in parallel threads |
| Thread       | Smallest execution unit on GPU         |
| Block        | Group of threads                       |
| Grid         | Group of blocks                        |
| `cudaMalloc` | Allocate GPU memory                    |
| `cudaMemcpy` | Copy memory between host and device    |
| `cudaFree`   | Free GPU memory                        |

---
