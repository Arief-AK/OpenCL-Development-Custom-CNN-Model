import numpy as np
import pyopencl as cl

from include.Logger import Logger
from include.Visualiser import Visualiser
from include.CL_CNNBuilder import CL_CNNBuilder

class Controller:
    def __init__(self):
        self.platform = cl.get_platforms()[0]
        self.device = self.platform.get_devices()[0]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context, properties=cl.command_queue_properties.PROFILING_ENABLE)
        
        self.BLOCK_SIZE = 32

        self.visualiser = Visualiser()
        self.logger = Logger(__name__)

    def visualise_model_layer(self, layer, shape):
        # Get feature maps (tensor) from buffers of layers
        layer_list = self.cnn.get_tensor(f"{layer}", shape)

        # Visualise each layer
        for tensor in layer_list:
            self.visualiser.visualise_feature_maps(tensor,num_filters=8)

    def cnn_model(self, image: np.ndarray, conv_kernel=None) -> tuple:
        image_width, image_height = image.shape

        if conv_kernel == None:
            conv_kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)  # Sobel-edge filter

        # Build a CNN model
        self.cnn = CL_CNNBuilder(self.context, self.queue, image_width, image_height, self.program, self.BLOCK_SIZE)
        self.cnn.conv2d(conv_kernel, 3).relu().max_pool(2)
        output = self.cnn.build(image)

        return output, self.cnn.profiling_info

    def bench_convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> tuple:
        # Validate inputs
        if image is None or kernel is None:
            raise ValueError("Image and kernel must not be None")
            
        image_width, image_height = image.shape
        kernel_size = kernel.shape[0]
        
        if kernel.shape[0] != kernel.shape[1]:
            raise ValueError("Kernel must be square")
            
        if image_width < kernel_size or image_height < kernel_size:
            raise ValueError("Image dimensions must be larger than kernel size")

        # Calculate output dimensions
        output_height = image_height - kernel_size + 1
        output_width = image_width - kernel_size + 1
        output = np.zeros((output_height, output_width), dtype=np.float32)

        # Create buffers with proper flags
        mf = cl.mem_flags
        try:
            image_buffer = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image)
            kernel_buffer = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=kernel)
            output_buffer = cl.Buffer(self.context, mf.WRITE_ONLY, output.nbytes)
        except cl.MemoryError:
            raise RuntimeError("Failed to allocate OpenCL buffers")

        # Set kernel arguments
        kernel_func = self.program.convolve
        kernel_func.set_arg(0, image_buffer)
        kernel_func.set_arg(1, kernel_buffer)
        kernel_func.set_arg(2, output_buffer)
        kernel_func.set_arg(3, np.int32(image_width))
        kernel_func.set_arg(4, np.int32(image_height))
        kernel_func.set_arg(5, np.int32(kernel_size))

        # Execute kernel with correct output size
        global_size = (output_width, output_height)
        event = cl.enqueue_nd_range_kernel(self.queue, kernel_func, global_size, None)
        event.wait()

        # Retrieve results
        cl.enqueue_copy(self.queue, output, output_buffer)
        self.queue.finish()

        # Measure execution time
        elapsed_time = (event.profile.end - event.profile.start) / 1e6
        return output, elapsed_time
    
    def bench_relu_activation(self, image:np.ndarray) -> tuple:
        image_width, image_height = image.shape
        size = image.size
        output = np.zeros_like(image)
        
        # Create buffers
        mf = cl.mem_flags
        input_buffer = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image)
        output_buffer = cl.Buffer(self.context, mf.WRITE_ONLY, output.nbytes)

        # Set kernel arguments
        kernel_func = self.program.relu_activation
        kernel_func.set_arg(0, input_buffer)
        kernel_func.set_arg(1, output_buffer)
        kernel_func.set_arg(2, np.int32(size))

        # Execute kernel with proper 2D size 
        global_size = (image_width, image_height)  # Maintain 2D structure
        local_size = (min(self.BLOCK_SIZE, image_width), min(self.BLOCK_SIZE, image_height))
        event = cl.enqueue_nd_range_kernel(self.queue, kernel_func, global_size, local_size)
        event.wait()

        # Retrieve results
        cl.enqueue_copy(self.queue, output, output_buffer)
        self.queue.finish()

        # Measure execution time
        elapsed_time = (event.profile.end - event.profile.start) / 1e6
        return output, elapsed_time
    
    def bench_max_pooling2d(self, image: np.ndarray, pool_size: int) -> tuple:
        image_width, image_height = image.shape
        output = np.zeros_like(image)

        # Create buffers
        mf = cl.mem_flags
        image_buffer = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image)
        output_buffer = cl.Buffer(self.context, mf.WRITE_ONLY, output.nbytes)

        # Set kernel arguments
        kernel_func = self.program.max_pooling
        kernel_func.set_arg(0, image_buffer)
        kernel_func.set_arg(1, output_buffer)
        kernel_func.set_arg(2, np.int32(image_width))
        kernel_func.set_arg(3, np.int32(image_height))
        kernel_func.set_arg(4, np.int32(pool_size))

        # Execute kernel
        global_size = (image_width, image_height)
        event = cl.enqueue_nd_range_kernel(self.queue, kernel_func, global_size, None)
        event.wait()

        # Retrieve results
        cl.enqueue_copy(self.queue, output, output_buffer)
        self.queue.finish()

        # Measure execution time
        elapsed_time = (event.profile.end - event.profile.start) / 1e6
        return output, elapsed_time
    
    def bench_dense(self, image: np.ndarray, weights:np.ndarray, bias:np.ndarray, input_size: int, output_size: int) -> tuple:
        output = np.zeros_like(image)

        # Create buffers
        mf = cl.mem_flags
        input_buffer = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image)
        weights_buffer = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weights)
        bias_buffer = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bias)
        output_buffer = cl.Buffer(self.context, mf.WRITE_ONLY, bias.nbytes)

        # Set kernel arguments
        kernel_func = self.program.dense
        kernel_func.set_arg(0, input_buffer)
        kernel_func.set_arg(1, weights_buffer)
        kernel_func.set_arg(2, bias_buffer)
        kernel_func.set_arg(3, output_buffer)
        kernel_func.set_arg(4, np.int32(input_size))
        kernel_func.set_arg(5, np.int32(output_size))

        # Execute kernel
        global_size = (output_size,)
        event = cl.enqueue_nd_range_kernel(self.queue, kernel_func, global_size, None)
        event.wait()

        # Retrieve results
        output = np.empty(output_size, dtype=np.float32)
        cl.enqueue_copy(self.queue, output, output_buffer)
        self.queue.finish()

        # Measure execution time
        elapsed_time = (event.profile.end - event.profile.start) / 1e6
        return output, elapsed_time

    def load_program(self, program_file: str):
        with open(program_file, 'r') as f:
            program_source = f.read()
        self.program = cl.Program(self.context, program_source).build(options=[f"-DBLOCK_SIZE={self.BLOCK_SIZE}"])

    def print_info(self):
        self.logger.info("OpenCL Information")

        self.logger.info("Platform Information")
        self._get_platform_info()
        
        self.logger.info("Device Information")
        self._get_device_info()

        self.logger.info("Context Information")
        self._get_context_info()

        self.logger.info("Queue Information")
        self._get_queue_info()

    def _get_platform_info(self):
        self.logger.info(f"\tPlatform: {self.platform.name}")
        self.logger.info(f"\tVendor: {self.platform.vendor}")
        self.logger.info(f"\tVersion: {self.platform.version}")

    def _get_device_info(self):
        self.logger.info(f"\tDevice: {self.device.name}")
        self.logger.info(f"\tType: {cl.device_type.to_string(self.device.type)}")
        self.logger.info(f"\tVersion: {self.device.version}")

    def _get_context_info(self):
        self.logger.info(f"\tContext: {self.context}")
        self.logger.info(f"\tDevices: {self.context.devices}")

    def _get_queue_info(self):
        self.logger.info(f"\tQueue: {self.queue}")
        self.logger.info(f"\tDevice: {self.queue.device}")

    def _get_program_info(self):
        self.logger.info(f"\tProgram: {self.program}")
        self.logger.info(f"\tDevices: {self.program.devices}")

    def get_platforms(self):
        return self.platform
    
    def get_devices(self):
        return self.device
    
    def get_contexts(self):
        return self.context
    
    def get_queues(self):
        return self.queue
    
    def get_programs(self):
        return self.program