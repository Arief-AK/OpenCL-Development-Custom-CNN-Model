#ifndef BLOCK_SIZE
    #define BLOCK_SIZE 32
#endif

__kernel void convolve(
    __global float* image,
    __global float* c_kernel,
    __global float* output,
    const int img_width,
    const int img_height,
    const int kernel_size)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    // Calculate output dimensions
    int out_width = img_width - kernel_size + 1;
    int out_height = img_height - kernel_size + 1;
    
    // Check bounds
    if (x >= out_width || y >= out_height) {
        return;
    }

    float sum = 0.0f;
    
    // Perform convolution
    for(int i = 0; i < kernel_size; i++) {
        for(int j = 0; j < kernel_size; j++) {
            sum += c_kernel[i * kernel_size + j] * 
                   image[(y + i) * img_width + (x + j)];
        }
    }

    output[y * out_width + x] = sum;
}