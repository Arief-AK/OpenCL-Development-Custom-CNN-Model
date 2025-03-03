import numpy as np
import pyopencl as cl

from include.Logger import Logger
from include.Comparator import Comparator
from include.Controller import Controller

IMAGE_SIZE = 1024

def cpu_convolve(comparator: Comparator) -> np.ndarray:
    image = np.random.rand(IMAGE_SIZE, IMAGE_SIZE)          # Random image
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) # Simple Sobel operator
    output = comparator.convolve2d(image, kernel)           # Perform convolution with Sobel-edge

    return output

def opencl_convolve(Controller: Controller):
    image = np.random.rand(IMAGE_SIZE, IMAGE_SIZE)          # Random image
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) # Simple Sobel operator

if __name__ == "__main__":
    # Create variables
    logger = Logger(__name__)
    comparator = Comparator()
    controller = Controller()

    # Perform convolution on CPU
    output = cpu_convolve(comparator)
    logger.info(f"CPU output: {output}")

    # Perform convolution on OpenCL
    controller.load_program("kernels/convolution.cl")
    
    # Display OpenCL information
    controller.print_info()

    # opencl_convolve(controller)