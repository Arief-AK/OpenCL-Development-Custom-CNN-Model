import numpy as np
import pyopencl as cl

from include.Logger import Logger
from include.Comparator import Comparator

def cpu_convolve(comparator: Comparator) -> np.ndarray:
    image = np.random.rand(1024, 1024)                      # Random image
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) # Simple Sobel operator
    output = comparator.convolve2d(image, kernel)           # Perform convolution with Sobel-edge

    return output

if __name__ == "__main__":
    # Create variables
    logger = Logger(__name__)
    comparator = Comparator()

    # Perform convolution on CPU
    output = cpu_convolve(comparator)
    logger.info(f"CPU output: {output}")