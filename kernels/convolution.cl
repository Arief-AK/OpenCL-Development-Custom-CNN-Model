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
    float sum = 0.0f;

    // Perform convolution within bounds
    if(x >= half_k && x < img_width - half_k && y >= half_k && y < img_height - half_k){
        for(int i = -half_k; i <= half_k; i++){
            for(int j = -half_k; j <= half_k; j++){
                int img_x = x + i;
                int img_y = y + j;
                int kernel_x = i + half_k;
                int kernel_y = j + half_k;
                
                sum += image[img_y * img_width + img_x] * c_kernel[kernel_y * kernel_size + kernel_x];
            }
        }
        output[y * img_width + x] = sum;
    }
}