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

__kernel void relu_activation(
    __global float* input,
    int size)
{
    int idx = get_global_id(0);
    if(idx < size){
        input[idx] = fmax(0.0f, input[idx]);
    }
}

__kernel void max_pooling(
    __global float* input,
    __global float* output,
    int width,
    int height,
    int pool_size)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    int half_p = pool_size / 2;

    // Perform max pooling within bounds
    float max_val = 0.0f;
    if(x < width && y < height){
        for(int i = 0; i < pool_size; i++){
            for(int j = 0; j < pool_size; j++){
                int image_x = clamp(x + i - half_p, 0, width - 1);
                int image_y = clamp(y + j - half_p, 0, height - 1);
                max_val = fmax(max_val, input[image_y * width + image_x]);
            }
        }
        output[y * width + x] = max_val;
    }
}

/**
* @brief Fully-connected (dense) layer.
         Based on the equation Y = WX + B

* @param input Input vector (X)
* @param weight Weight matrix (W)
* @param bias Bias vector (B)
* @param output Output vector (Y)
* @param input_size size of the input vector
* @param output_size size of the output vector

*/
__kernel void dense(
    __global float* input,
    __global float* weights,
    __global float* bias,
    __global float* output,
    int input_size,
    int output_size)
{
    // Work-item computes one neuron in layer
    int neuron_idx = get_global_id(0);

    if(neuron_idx < output_size){
        // Initialise with bias
        float sum = bias[neuron_idx];

        // Calculate weighted sum with bias
        for(int i = 0; i < input_size; i++){
            sum += weights[neuron_idx * input_size + i] * input[i];
        }

        // Store in the output buffer
        output[neuron_idx] = sum;
    }
}