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