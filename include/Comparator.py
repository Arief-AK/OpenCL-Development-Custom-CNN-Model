import time
import numpy as np
from scipy.signal import correlate2d

from include.Logger import Logger

class Comparator:
    def __init__(self):
        self.logger = Logger(__name__)

    def convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> tuple:
        self.logger.debug("Performing 2D convolution")
        
        start_time = time.time()
        height, width = image.shape
        k_height, k_width = kernel.shape
        half_kh, half_kw = k_height // 2, k_width // 2
        
        # Calculate output dimensions
        out_height = height - k_height + 1
        out_width = width - k_width + 1
        
        # Initialize output with correct size
        output = np.zeros((out_height, out_width), dtype=np.float32)
        
        # Manual convolution with correct output size
        for y in range(out_height):
            for x in range(out_width):
                sum_val = 0.0
                for i in range(k_height):
                    for j in range(k_width):
                        sum_val += kernel[i, j] * image[y + i, x + j]
                output[y, x] = sum_val
        
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        return output, elapsed_time
    
    def relu_activation(self, image:np.ndarray) -> tuple:
        self.logger.debug("Performing ReLU activation")
        
        start_time = time.time()
        output = np.maximum(image, 0.0)
        end_time = time.time()

        elapsed_time = (end_time - start_time) * 1000
        return output, elapsed_time
    
    def max_pooling2d(self, image:np.ndarray, pool_size: int) -> tuple:
        self.logger.debug("Performing 2D max pooling")
        
        start_time = time.time()
        height, width = image.shape
        half_p = pool_size // 2
        
        # Output array (same size as input for now)
        output = np.zeros((height, width), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                max_val = 0.0
                
                # Apply max pooling within bounds
                for i in range(pool_size):
                    for j in range(pool_size):
                        image_x = np.clip(x + i - half_p, 0, width - 1)
                        image_y = np.clip(y + j - half_p, 0, height - 1)
                        max_val = max(max_val, image[image_y, image_x])
                
                output[y, x] = max_val
        end_time = time.time()
        
        elapsed_time = (end_time - start_time) * 1000
        return output, elapsed_time
    
    def dense(self, image: np.ndarray, weights:np.ndarray, bias:np.ndarray) -> tuple:
        self.logger.debug("Performing dense layer")

        start_time = time.time()
        output = np.dot(weights, image) + bias
        end_time = time.time()

        elapsed_time = (end_time - start_time) * 1000
        return output, elapsed_time