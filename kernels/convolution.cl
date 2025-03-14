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

    int half_k = kernel_size / 2;

    // Process valid regions
    if(x >= half_k && x < img_width - half_k && y >= img_height - half_k){
        float sum = 0.0f;

        // Convolve
        for(int i = 0; i < kernel_size; i++){
            for(int j = 0; j < kernel_size; j++){
                int image_x = x + i - half_k;
                int image_y = y + j - half_k;
                sum += c_kernel[i * kernel_size +j] * image[image_y * img_width + image_x];
            }
        }
    
        // Store in valid output region
        int valid_x = x - half_k;
        int valid_y = y - half_k;
        int valid_width = img_width - kernel_size + 1;
        output[valid_y * valid_width + valid_x] = sum;
    }
}