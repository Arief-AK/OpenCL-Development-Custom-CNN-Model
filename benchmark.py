import numpy as np

from include.Logger import Logger
from include.Comparator import Comparator
from include.Controller import Controller

IMAGE_SIZE = 512

def Benchmark(controller: Controller, comparator: Comparator, function:str, logger:Logger):
    # Initialise variables
    image = np.random.rand(IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)   # Random image

    # Perform function on CPU and OpenCL
    if function == "Convolution":
        logger.info("Performing 2D convolution")
        controller.load_program("kernels/convolution.cl")

        kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])         # Simple Sobel operator
        cpu_output, cpu_time = comparator.convolve2d(image, kernel)
        logger.debug(f"Performed {function} on CPU")
        opencl_output, opencl_time = controller.bench_convolve2d(image, kernel)
        logger.debug(f"Performed {function} with OpenCL")
    
    elif function == "ReLU":
        logger.info("Perfoming ReLU activation")
        controller.load_program("kernels/relu.cl")

        cpu_output, cpu_time = comparator.relu_activation(image)
        logger.debug(f"Performed {function} on CPU")
        opencl_output, opencl_time = controller.bench_relu_activation(image)
        logger.debug(f"Performed {function} with OpenCL")

    elif function == "MaxPooling":
        logger.info("Performing 2D max pooling")
        controller.load_program("kernels/max_pooling.cl")

        cpu_output, cpu_time = comparator.max_pooling2d(image, 2)
        logger.debug(f"Performed {function} on CPU")
        opencl_output, opencl_time = controller.bench_max_pooling2d(image, 2)
        logger.debug(f"Performed {function} with OpenCL")

    else:
        logger.error("Invalid function")
        return exit(1)

    # Profile timings
    logger.info(f"********* {function} Summary *********")
    logger.info(f"CPU execution time: {cpu_time:.2f} ms")
    logger.info(f"OpenCL execution time: {opencl_time:.2f} ms")

if __name__ == "__main__":
    # Create variables
    logger = Logger(__name__)
    comparator = Comparator()
    controller = Controller()
    controller.print_info()

    # Perform benchmarking
    logger.info(f"************ START OF BENCHMARKING ************")
    Benchmark(controller, comparator, "Convolution", logger)
    Benchmark(controller, comparator, "ReLU", logger)
    Benchmark(controller, comparator, "MaxPooling", logger)
    logger.info(f"************ END OF BENCHMARKING ************")