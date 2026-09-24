#include "../common/book.h"

#define N (33 * 1024)

__global__ void add (int * a, int * b, int * c) {
    // the "tid" word here is confusing. Every thread (with its unique tid) will do operations on different
    // positions (indexes) of the memory.
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    while (index < N) {
        c[index] = a[index] + b[index];
        index += blockDim.x * gridDim.x;
    }
}

int main (void) {
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    // Allocate memory on the GPU
    HANDLE_ERROR(cudaMalloc ((void**) &dev_a, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc ((void**) &dev_b, N * sizeof(int)));
    HANDLE_ERROR(cudaMalloc ((void**) &dev_c, N * sizeof(int)));

    // Fill the arrays a and b with N elements
    for (int i = 0; i < N; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }

    // Copy the arrays a and b into the GPU
    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

    add<<<128,128>>>(dev_a, dev_b, dev_c);

    // Copy the array C back from GPU to CPU
    HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

    // verify that the GPU did the work we requested
    bool success = true;
    for (int i=0; i<N; i++) {
        if ((a[i] + b[i]) != c[i]) {
            printf( "Error: %d + %d != %d\n", a[i], b[i], c[i] );
            success = false;
        }   
    }
    if (success) printf( "We did it!\n" );

    // Free the memory allocated in the GPU
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}



