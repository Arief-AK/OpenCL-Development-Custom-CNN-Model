import numpy as np

from include.Logger import Logger
from include.Comparator import Comparator
from include.Controller import Controller

IMAGE_SIZE = 512

def compare(opencl_output: np.ndarray, cpu_output: np.ndarray, logger: Logger):
    if np.isnan(opencl_output).any() or np.isinf(opencl_output).any():
        logger.warning("⚠️ Warning: OpenCL output contains NaNs or Infs!")
    else:
        if np.allclose(cpu_output, opencl_output, atol=1e-5):
            logger.info("✅ OpenCL and CPU results match closely!")
        else:
            logger.error("❌ Significant differences detected!")

def Benchmark(controller: Controller, comparator: Comparator, function:str, logger:Logger):
    # Initialise variables
    #image = np.random.rand(IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)   # Random image
    image = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25]
    ], dtype=np.float32)

    # Perform function on CPU and OpenCL
    if function == "Convolution":
        logger.info("Performing 2D convolution")
        controller.load_program("kernels/convolution.cl")

        kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])         # Simple Sobel operator
        cpu_output, cpu_time = comparator.convolve2d(image, kernel)
        logger.debug(f"Performed {function} on CPU")
        opencl_output, opencl_time = controller.bench_convolve2d(image, kernel)
        logger.debug(f"Performed {function} with OpenCL")

        compare(opencl_output, cpu_output, logger)
    
    elif function == "ReLU":
        logger.info("Perfoming ReLU activation")
        controller.load_program("kernels/relu.cl")

        cpu_output, cpu_time = comparator.relu_activation(image)
        logger.debug(f"Performed {function} on CPU")
        opencl_output, opencl_time = controller.bench_relu_activation(image)
        logger.debug(f"Performed {function} with OpenCL")

        compare(opencl_output, cpu_output, logger)

    elif function == "MaxPooling":
        logger.info("Performing 2D max pooling")
        controller.load_program("kernels/max_pooling.cl")

        cpu_output, cpu_time = comparator.max_pooling2d(image, 2)
        logger.debug(f"Performed {function} on CPU")
        opencl_output, opencl_time = controller.bench_max_pooling2d(image, 2)
        logger.debug(f"Performed {function} with OpenCL")

        compare(opencl_output, cpu_output, logger)

    elif function == "Dense":
        logger.info("Performing Dense")
        controller.load_program("kernels/dense.cl")

        # Random data
        input_size = 5
        output_size = 3
        input_data = np.random.rand(input_size).astype(np.float32)
        weights = np.random.rand(output_size, input_size).astype(np.float32)
        bias = np.random.rand(output_size).astype(np.float32)

        cpu_output, cpu_time = comparator.dense(input_data, weights, bias)
        logger.info(f"Performed {function} on CPU")
        opencl_output, opencl_time = controller.bench_dense(input_data, weights, bias, input_size, output_size)
        logger.info(f"Peformed {function} with OpenCL")

        compare(opencl_output, cpu_output, logger)

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
    Benchmark(controller, comparator, "Dense", logger)
    logger.info(f"************ END OF BENCHMARKING ************")