import numpy as np

from include.Logger import Logger
from include.Comparator import Comparator
from include.Controller import Controller

IMAGE_SIZE = 1024

def cpu_convolve(comparator: Comparator) -> np.ndarray:
    image = np.random.rand(IMAGE_SIZE, IMAGE_SIZE)          # Random image
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) # Simple Sobel operator
    output = comparator.convolve2d(image, kernel)           # Perform convolution with Sobel-edge

    return output

def opencl_convolve(controller: Controller) -> tuple:
    image = np.random.rand(IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)           # Random image
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)   # Simple Sobel operator

    return controller.convolve2d(image, kernel)

if __name__ == "__main__":
    # Create variables
    logger = Logger(__name__)
    comparator = Comparator()
    controller = Controller()

    # Perform convolution on CPU
    cpu_output, cpu_time = cpu_convolve(comparator)

    # Perform convolution on OpenCL
    controller.load_program("kernels/convolution.cl")
    
    # Display OpenCL information
    controller.print_info()

    # Perform convolution on OpenCL
    opencl_output, opencl_time = opencl_convolve(controller)

    # Profile timings
    logger.info(f"CPU execution time: {cpu_time:.2f} ms")
    logger.info(f"OpenCL execution time: {opencl_time:.2f} ms")