import numpy as np

from include.Logger import Logger
from include.Comparator import Comparator
from include.Controller import Controller

IMAGE_SIZE = 512
SHOW_DEBUG_MSG = True

def validate_input(array: np.ndarray, name: str, logger: Logger) -> bool:
    if array is None:
        logger.error(f"{name} is None")
        return False
    if not isinstance(array, np.ndarray):
        logger.error(f"{name} must be a numpy array")
        return False
    if not array.dtype == np.float32:
        logger.error(f"{name} must be float32, got {array.dtype}")
        return False
    if np.isnan(array).any() or np.isinf(array).any():
        logger.error(f"{name} contains NaN or Inf values")
        return False
    return True

def compare(opencl_output: np.ndarray, cpu_output: np.ndarray, logger: Logger):
    if np.isnan(opencl_output).any() or np.isinf(opencl_output).any():
        logger.warning("⚠️ Warning: OpenCL output contains NaNs or Infs!")
    else:
        if np.allclose(cpu_output, opencl_output, atol=1e-5):
            logger.info("✅ OpenCL and CPU results match closely!")
        else:
            logger.error("❌ Significant differences detected!")

def Benchmark(controller: Controller, comparator: Comparator, function:str, logger:Logger):
    # Initialise variables with explicit float32 type and validation
    image = np.random.rand(IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)

    if not validate_input(image, "image", logger):
        return

    indices = [(0, 0), (0,1) , (1, 0), (1, 1), (2, 0), (2, 1)]

    # Perform function on CPU and OpenCL
    if function == "Convolution":
        logger.info("Performing 2D convolution")
        controller.load_program("kernels/convolution.cl")

        kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
        if not validate_input(kernel, "kernel", logger):
            return

        # Log array shapes and data types
        if SHOW_DEBUG_MSG:
            logger.debug(f"Image shape: {image.shape}, dtype: {image.dtype}")
            logger.debug(f"Kernel shape: {kernel.shape}, dtype: {kernel.dtype}")

        cpu_output, cpu_time = comparator.convolve2d(image, kernel)
        logger.info(f"Performed {function} on CPU")
        opencl_output, opencl_time = controller.bench_convolve2d(image, kernel)
        logger.info(f"Performed {function} with OpenCL")

        if SHOW_DEBUG_MSG:
            for idx in indices:
                i, j = idx
                logger.debug(f"CPU[{i}, {j}]: {cpu_output[i, j]:.4f}, OpenCL[{i}, {j}]: {opencl_output[i, j]:.4f}")

        compare(opencl_output, cpu_output, logger)
    
    elif function == "ReLU":
        logger.info("Perfoming ReLU activation")
        controller.load_program("kernels/relu.cl")

        if SHOW_DEBUG_MSG:
            logger.debug(f"Image shape: {image.shape}, dtype: {image.dtype}")

        cpu_output, cpu_time = comparator.relu_activation(image)
        logger.info(f"Performed {function} on CPU")
        opencl_output, opencl_time = controller.bench_relu_activation(image)
        logger.info(f"Performed {function} with OpenCL")

        if SHOW_DEBUG_MSG:
            for idx in indices:
                i, j = idx
                logger.debug(f"CPU[{i}, {j}]: {cpu_output[i, j]:.4f}, OpenCL[{i}, {j}]: {opencl_output[i, j]:.4f}")

        compare(opencl_output, cpu_output, logger)

    elif function == "MaxPooling":
        logger.info("Performing 2D max pooling")
        controller.load_program("kernels/max_pooling.cl")

        if SHOW_DEBUG_MSG:
            logger.debug(f"Image shape: {image.shape}, dtype: {image.dtype}")

        cpu_output, cpu_time = comparator.max_pooling2d(image, 2)
        logger.debug(f"Performed {function} on CPU")
        opencl_output, opencl_time = controller.bench_max_pooling2d(image, 2)
        logger.debug(f"Performed {function} with OpenCL")

        if SHOW_DEBUG_MSG:
            for idx in indices:
                i, j = idx
                logger.debug(f"CPU[{i}, {j}]: {cpu_output[i, j]:.4f}, OpenCL[{i}, {j}]: {opencl_output[i, j]:.4f}")

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