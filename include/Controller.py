import time
import numpy as np
import pyopencl as cl

from include.Logger import Logger

class Controller:
    def __init__(self):
        self.platform = cl.get_platforms()[0]
        self.device = self.platform.get_devices()[0]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context, properties=cl.command_queue_properties.PROFILING_ENABLE)
        self.logger = Logger(__name__)

    def convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        image_width, image_height = image.shape
        kernel_size = kernel.shape[0]
        output = np.zeros_like(image)

        # Create buffers
        mf = cl.mem_flags
        image_buffer = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image)
        kernel_buffer = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=kernel)
        output_buffer = cl.Buffer(self.context, mf.WRITE_ONLY, output.nbytes)

        # Set kernel arguments
        kernel_func = self.program.convolve
        kernel_func.set_arg(0, image_buffer)
        kernel_func.set_arg(1, kernel_buffer)
        kernel_func.set_arg(2, output_buffer)
        kernel_func.set_arg(3, np.int32(image_width))
        kernel_func.set_arg(4, np.int32(image_height))
        kernel_func.set_arg(5, np.int32(kernel_size))

        # Execute kernel
        global_size = (image_width, image_height)
        event = cl.enqueue_nd_range_kernel(self.queue, kernel_func, global_size, None)
        event.wait()

        # Retrieve results
        cl.enqueue_copy(self.queue, output, output_buffer)
        self.queue.finish()

        # Measure execution time
        elapsed_time = (event.profile.end - event.profile.start) / 1e6
        self.logger.info(f"Execution time: {elapsed_time} ms")

    def load_program(self, program_file: str):
        with open(program_file, 'r') as f:
            program_source = f.read()
        self.program = cl.Program(self.context, program_source).build()

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

        self.logger.info("Program Information")
        self._get_program_info()

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