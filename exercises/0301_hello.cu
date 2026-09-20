#include <stdio.h>

__global__ void helloFromGPU() {
    printf("Hello World from GPU! Thread:%d\n", threadIdx.x);
}

int main() {
    helloFromGPU<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}