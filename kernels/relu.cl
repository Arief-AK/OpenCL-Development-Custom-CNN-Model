// Rectified Linear Unit (ReLU) activation function
// f(x) = max(0, x)
// Takes an input tensor and modifies it in-place

__kernel void relu_activation(
    __global float* input,
    int size)
{
    int idx = get_global_id(0);
    if(idx < size){
        input[idx] = fmax(0.0f, input[idx]);
    }
}