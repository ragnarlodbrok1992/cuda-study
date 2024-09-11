#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

int main() {
    int device_count;
    int device;
    unsigned int device_flags;

    // Collect device information
    cudaGetDeviceCount(&device_count);
    cudaGetDevice(&device);
    cudaGetDeviceFlags(&device_flags);

    // Information
    printf("Device count: %d\n", device_count);
    printf("Device used:  %d\n", device);
    printf("Device flags: 0x%08x\n", device_flags);

    return 0;
}

