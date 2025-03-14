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

__kernel void relu_activation(
    __global float* input,
    __global float* output,
    const int size)
{
    // Get 2D position of the thread
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width = get_global_size(0);
    
    // Calculate linear index from 2D position
    int idx = y * width + x;
    
    if(idx < size) {
        output[idx] = fmax(0.0f, input[idx]);
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