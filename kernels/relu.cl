// Rectified Linear Unit (ReLU) activation function
// f(x) = max(0, x)
// Takes an input tensor and modifies it in-place

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