// Average pooling kernel
// Function takes the average value from a given window
// Each work-item processes a single output pixel by scanning its corresponding region

__kernel void avg_pooling(
    __global float* input,
    __global float* output,
    int width,
    int height,
    int pool_size)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    int half_p = pool_size / 2;

    // Perform average pooling within bounds
    float sum_val = 0.0f;
    if(x < width && y < height){
        for(int i = 0; i < pool_size; i++){
            for(int j = 0; j < pool_size; j++){
                int image_x = clamp(x + i - half_p, 0, width - 1);
                int image_y = clamp(y + j - half_p, 0, height - 1);
                sum_val += input[image_y * width + image_x];
            }
        }
        output[y * width + x] = sum_val / (pool_size * pool_size);
    }
}