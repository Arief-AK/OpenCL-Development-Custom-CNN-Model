// Max Pooling kernel
// Function takes the maximum value from a given window
// Each work-item processes a single output pixel by scanning its corresponding region

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