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

    // Defined local tile memory
    __local float local_tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    // Get local coordinates
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);
    
    int half_k = kernel_size / 2;
    
    // Load data into local tile
    int image_x = clamp(x - half_k, 0, img_width - 1);
    int image_y = clamp(y - half_k, 0, img_height - 1);
    local_tile[local_y][local_x] = image[image_y * img_width + image_x];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform convolution within bounds
    float sum = 0.0f;
    if(local_x < BLOCK_SIZE && local_y < BLOCK_SIZE){
        for(int i = 0; i < kernel_size; i++){
            for(int j = 0; j < kernel_size; j++){
                int image_x = clamp(x + i - half_k, 0, img_width - 1);
                int image_y = clamp(y + j - half_k, 0, img_height - 1);
                sum += c_kernel[i * kernel_size + j] * image[image_y * img_width + image_x];
            }
        }
        output[y * img_width + x] = sum;
    }
}