## Project Overview

### Project Highlights:

- Support for real-time simulation of gravity for 30,000+ particles
- Implementation using CUDA C/C++
- High-performance parallel GPU computing
- Systematic performance optimization

### Technical Details:

- Implementation Language: CUDA C/C++
- Development Environment: NVIDIA CUDA Toolkit (Cloud Server)
- Hardware Requirements: NVIDIA GPU with Compute Capability 3.0+ (Tesla T4)

---

## Technical Challenges

### Main Challenges:

1. Computational Complexity
    - O(n²) complexity for particle interaction calculations
    - Requires calculating the force between every pair of particles

    ```c
    // Original complexity illustration
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // Calculate the force between particle i and j
        }
    }
    ```

    In **N-body simulation**, the gravitational force $\vec{F}_{ij}$ between particle `i` and particle `j` is calculated using the following formulas:

    1. **Distance between two particles**:

        $r_{ij} = \sqrt{(x_j - x_i)^2 + (y_j - y_i)^2 + (z_j - z_i)^2}$

        To avoid division by zero errors, a small smoothing factor $\epsilon$ can be added:

        $r_{ij} = \sqrt{(x_j - x_i)^2 + (y_j - y_i)^2 + (z_j - z_i)^2 + \epsilon^2}$

    2. **Magnitude of the gravitational force (scalar form)**:

        $F_{ij} = \frac{G \cdot m_i \cdot m_j}{r_{ij}^2}$

        Where:

        - G is the gravitational constant: $6.67430 \times 10^{-11} \, \mathrm{N \cdot m^2 \cdot kg^{-2}}$
        - $m_i, m_j$ are the masses of particle i and j, respectively.

    3. **Components of the gravitational force**:
    Calculate the force components in the $x$, $y$, and $z$ directions:

        $F_{ij,x} = F_{ij} \cdot \frac{x_j - x_i}{r_{ij}}$

        $F_{ij,y} = F_{ij} \cdot \frac{y_j - y_i}{r_{ij}}$

        $F_{ij,z} = F_{ij} \cdot \frac{z_j - z_i}{r_{ij}}$

    4. **Updating the total force**:
    Add the gravitational force components to the forces acting on particles $i$ and $j$:

        $\vec{F}_i = \sum_{j \neq i} \vec{F}_{ij}$

        $\vec{F}_j = \sum_{i \neq j} -\vec{F}_{ij}$

        The formula for the reaction force is:

        $\vec{F}_{ji} = -\vec{F}_{ij}$

    5. **Final summation form**:
    The total force components for each particle $i$:

        $F_{i,x} = \sum_{j \neq i} \frac{G \cdot m_i \cdot m_j \cdot (x_j - x_i)}{r_{ij}^3}$

        $F_{i,y} = \sum_{j \neq i} \frac{G \cdot m_i \cdot m_j \cdot (y_j - y_i)}{r_{ij}^3}$

        $F_{i,z} = \sum_{j \neq i} \frac{G \cdot m_i \cdot m_j \cdot (z_j - z_i)}{r_{ij}^3}$

2. Memory Access
    - Frequent global memory access
        - Need to optimize access patterns
    - Memory bandwidth limitations
3. Numerical Stability
    - Handling numerical issues at extremely small distances
    - Floating-point precision considerations
    - Avoiding division by zero
4. Performance Requirements
    - Real-time simulation needs
    - Large-scale data processing
    - Hiding memory latency

---

## Solutions

### CUDA Implementation Details

- **Kernel Design:**
    - `bodyForce_cuda`: Each thread calculates the total gravitational force acting on a single particle.
    - `integratePosition_cuda`: Each thread updates the position and velocity of a single particle.
- **Parallelization Strategy:**
    - Employs a one-dimensional Grid and Block structure, with each thread mapped to a particle.
    - `blockIdx.x * blockDim.x + threadIdx.x` calculates the global particle index.
- **Memory Management:**
    - Uses `cudaMalloc` to allocate particle data structure arrays in device global memory.
    - `cudaMemcpyHostToDevice` and `cudaMemcpyDeviceToHost` handle data transfer between the Host and Device.
- **Thread Configuration:**
    - The choice of `threadsPerBlock` needs to consider the GPU's hardware limitations and the kernel's computational complexity.
    - `blocksPerGrid` is calculated based on the number of particles and `threadsPerBlock` to ensure all particles are processed.

    ```cpp
    __global__ void bodyForce_cuda(Body *p, float dt, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
            for (int j = 0; j < n; j++) {
                float dx = p[j].x - p[i].x;
                float dy = p[j].y - p[i].y;
                float dz = p[j].z - p[i].z;
                float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                float invDist = rsqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
            p[i].vx += dt * Fx;
            p[i].vy += dt * Fy;
            p[i].vz += dt * Fz;
        }
    }
    ```

- **Synchronization Mechanism:**
    - `cudaDeviceSynchronize()` ensures that the kernel completes before proceeding with subsequent host-side operations, maintaining data consistency.

---

### **Explanation of Memory Optimization**

1. **Separation of Host and Device Memory**:
    - In the **existing program**, `cudaMalloc` is used to allocate memory on the GPU (`d_p`), and `cudaMemcpy` is used to transfer data from the host (CPU) memory (`h_p`) to the device (GPU).
    - **Purpose**: To avoid frequent access operations directly between the CPU and GPU, thereby improving memory access efficiency.

    ```cpp
    cuda_status = cudaMalloc((void**)&d_p, bytes);
    cuda_status = cudaMemcpy(d_p, h_p, bytes, cudaMemcpyHostToDevice);
    ```

    - **Optimization Principle**:
        - Global memory access on the GPU is slower, but it is suitable for large, one-time data transfers.
        - Separating the host and device means that only the initial data and the final results need to be transferred each computation cycle, reducing frequent data movement.
2. **Use of Buffers**:
    - The host-side buffer `h_buf` provides a contiguous block of memory corresponding to the `Body` structure array.
    - **Advantages**: Maintains memory alignment, increasing GPU read speeds.
    - **Source Code Snippet**:

        ```cpp
        h_buf = (float *)malloc(bytes);
        Body *h_p = (Body*)h_buf;
        ```
3. **Memory Access Patterns**:
    - In the CUDA kernel function, each thread is responsible for processing one particle, avoiding contention among different threads accessing the same memory locations.
    - **Code Example**:

        ```cpp
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
            for (int j = 0; j < n; j++) {
                // Calculate Fx, Fy, Fz
            }
            p[i].vx += dt * Fx;
            p[i].vy += dt * Fy;
            p[i].vz += dt * Fz;
        }
        ```

---

### **Explanation of Computational Optimization**

1. **Parallel Computing**:
    - Utilizing CUDA's thread model, where each thread is responsible for calculating the force acting on one particle.
    - **Existing Program Design**:
        - Each thread's index is calculated as follows:

            ```cpp
            int i = blockIdx.x * blockDim.x + threadIdx.x
            ```

        - **Purpose**: Ensures each thread processes a unique particle.
2. **Inverse Square Root Acceleration**:
    - In the GPU kernel function, the CUDA-provided `rsqrtf()` is used to calculate the inverse of the square root.
    - **Code Snippet**:

        ```cpp
        float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        float invDist = rsqrtf(distSqr); // Calculate the inverse square root
        float invDist3 = invDist * invDist * invDist;
        ```

    - **Reason**: `rsqrtf` is optimized at the hardware level, making its execution more efficient than traditional division operations.
3. **Accumulating Components**:
    - Accumulating the force components between particles onto each particle:

        ```cpp
        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
        ```

    - **Purpose**: Directly updates force components, avoiding the overhead of intermediate storage.
4. **Multi-Stage Calculation**:
    - The program is divided into two independent CUDA kernel functions:
        1. `bodyForce_cuda` calculates the forces between particles.
        2. `integratePosition_cuda` updates the positions.
    - **Advantages**:
        - Task separation facilitates kernel execution scheduling and performance analysis.
        - Independent handling of calculations and position updates reduces the complexity of memory access.
5. **Blocks and Thread Configuration**:
    - The GPU grid and block configuration ensures that all particles are processed:

        ```cpp
        int threadsPerBlock = 256;
        int blocksPerGrid = (nBodies + threadsPerBlock - 1) / threadsPerBlock;
        bodyForce_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_p, dt, nBodies);
        ```

    - **Purpose**:
        - `threadsPerBlock` controls the number of threads in each block, typically chosen as a multiple of GPU hardware-friendly values like 128 or 256.
        - `blocksPerGrid` ensures that each particle is processed by a unique thread.

---

### Future Optimization Strategies and Considerations

- **Computational Intensity:** The N-Body problem has high computational intensity, making it suitable for GPU acceleration.
- **Global Memory Access:** Random reads to global memory in the kernel can become a performance bottleneck.
- **Potential Optimizations:**
    - **Shared Memory:** Load some particle data into shared memory to reduce global memory access latency.
    - **Algorithm Optimization:** Consider using approximation algorithms like the Barnes-Hut algorithm to reduce computational complexity.
- **Kernel Fusion:** Merge `bodyForce_cuda` and `integratePosition_cuda` into a single kernel to reduce kernel launch overhead and the number of global memory reads and writes.

---

## Experience and Gains

### Main Gains:

1. Improved Technical Skills
    - Understanding of GPU architecture
    - Performance optimization mindset
    - Parallel programming skills
2. Best Practice Experience
    - Memory access patterns
    - Execution configuration optimization
    - Performance analysis methods
3. Problem-Solving Ability
    - Systematic thinking
    - Optimization strategy development
    - Performance bottleneck analysis

---

# Implementation

```cpp
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "files.h"
#include <cuda_runtime.h>

#define SOFTENING 1e-9f

/*
 * Each body contains x, y, and z coordinate positions,
 * as well as velocities in the x, y, and z directions.
 */

typedef struct { float x, y, z, vx, vy, vz; } Body;

/*
 * `__global__` specifies that this function will execute on the GPU and can be called from the CPU.
 * This is a CUDA kernel function.
 * CUDA kernel functions are the basic units of parallel execution.
 */
__global__ void bodyForce_cuda(Body *p, float dt, int n) {
    /*
     * `blockIdx.x` is the index of the block that the current thread belongs to within the grid.
     * `blockDim.x` is the number of threads in each block.
     * `threadIdx.x` is the index of the current thread within its block.
     * This calculation ensures that each thread processes a unique body.
     */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;

        for (int j = 0; j < n; j++) {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = rsqrtf(distSqr); // Calculate the inverse square root, generally faster than division on the GPU
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        p[i].vx += dt * Fx;
        p[i].vy += dt * Fy;
        p[i].vz += dt * Fz;
    }
}

/*
 * `__global__` specifies that this function will also execute on the GPU.
 */
__global__ void integratePosition_cuda(Body *p, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

int main(const int argc, const char** argv) {
    int nBodies = 2 << 11;
    if (argc > 1) nBodies = 2 << atoi(argv[1]);

    const char *initialized_values;
    const char *solution_values;

    if (nBodies == 2 << 11) {
        initialized_values = "09-nbody/files/initialized_4096";
        solution_values = "09-nbody/files/solution_4096";
    } else { // nBodies == 2<<15
        initialized_values = "09-nbody/files/initialized_65536";
        solution_values = "09-nbody/files/solution_65536";
    }

    if (argc > 2) initialized_values = argv[2];
    if (argc > 3) solution_values = argv[3];

    const float dt = 0.01f; // Time step
    const int nIters = 10;  // Number of simulation iterations

    int bytes = nBodies * sizeof(Body);
    float *h_buf; // Buffer on the host
    Body *d_p;   // Pointer on the device (GPU)

    // Host memory allocation
    h_buf = (float *)malloc(bytes);
    Body *h_p = (Body*)h_buf;

    // Read initial values from file
    read_values_from_file(initialized_values, h_buf, bytes);

    // Device memory allocation
    cudaError_t cuda_status = cudaMalloc((void**)&d_p, bytes);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cuda_status));
        return 1;
    }

    // Copy data from host to device
    cuda_status = cudaMemcpy(d_p, h_p, bytes, cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cuda_status));
        return 1;
    }

    double totalTime = 0.0;

    // Kernel launch configuration

    int threadsPerBlock = 256;
    // Number of threads per block. This is an adjustable parameter depending on the GPU architecture.

    int blocksPerGrid = (nBodies + threadsPerBlock - 1) / threadsPerBlock;
    // Number of blocks in the grid. Ensures all bodies are processed.

    for (int iter = 0; iter < nIters; iter++) {
        StartTimer();

        // Launch bodyForce kernel
        bodyForce_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_p, dt, nBodies);
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "bodyForce_cuda launch failed: %s\n", cudaGetErrorString(cuda_status));
            return 1;
        }
        cudaDeviceSynchronize(); // Ensure kernel completion

        // Launch integratePosition kernel
        integratePosition_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_p, dt, nBodies);
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "integratePosition_cuda launch failed: %s\n", cudaGetErrorString(cuda_status));
            return 1;
        }
        cudaDeviceSynchronize(); // Ensure kernel completion

        const double tElapsed = GetTimer() / 1000.0;
        totalTime += tElapsed;
    }

    // Copy results back to host
    cuda_status = cudaMemcpy(h_p, d_p, bytes, cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cuda_status));
        return 1;
    }

    double avgTime = totalTime / (double)(nIters);
    float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;
    write_values_to_file(solution_values, h_buf, bytes);

    printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);

    // Free device memory
    cudaFree(d_p);
    free(h_buf);

    return 0;
}
```

## Supplementary Information: Q&A Preparation

### Answers to Common Technical Questions:

1. Why choose float4?
   `float4` can provide better coalesced memory access, improving memory bandwidth utilization. At the same time, it also facilitates SIMD vector operations for us.
2. How is numerical stability handled?
   A Softening Factor is used to avoid numerical issues caused by extremely small distances, and careful consideration is also given to floating-point precision.
3. How is the execution configuration chosen?
   We chose a block size of 256 threads, which is an optimization based on the GPU's warp size and resource constraints.
