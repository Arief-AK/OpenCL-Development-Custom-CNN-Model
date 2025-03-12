import numpy as np
import pyopencl as cl

from include.Logger import Logger

class CL_CNNBuilder:
    def __init__(self, context, queue, width, height, program, block_size=32):
        self.context = context
        self.queue = queue
        self.width = width
        self.height = height
        self.program = program
        self.block_size = block_size

        self.layers = []
        self.buffers = []
        self.profiling_info = {}

        self.logger = Logger(__name__)

    def _create_buffer(self, shape):
        # Create an OpenCL buffer for given shape
        mf = cl.mem_flags
        return cl.Buffer(self.context, mf.READ_WRITE, size=np.prod(shape) * 4)  # Using float32 = 4 bytes
    
    def _record_time(self, event, layer_name):
        # Extract profiling time
        self.queue.finish()
        exec_time = (event.profile.end - event.profile.start) / 1e6
        self.profiling_info.update({layer_name:exec_time})

    def conv2d(self, kernel, kernel_size):
        # Add convolution layer
        def conv_layer(input_buffer):
            mf = cl.mem_flags
            
            # Create buffers
            output_buffer = self._create_buffer((self.height, self.width))
            kernel_buffer = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=kernel)

            # Set kernel arguments
            kernel_func = self.program.convolve
            kernel_func.set_arg(0, input_buffer)
            kernel_func.set_arg(1, kernel_buffer)
            kernel_func.set_arg(2, output_buffer)
            kernel_func.set_arg(3, np.int32(self.width))
            kernel_func.set_arg(4, np.int32(self.height))
            kernel_func.set_arg(5, np.int32(kernel_size))

            # Execute kernel
            global_size = (self.width, self.height)
            event = cl.enqueue_nd_range_kernel(self.queue, kernel_func, global_size, None)
            event.wait()

            self._record_time(event, "Convolution")
            return output_buffer
        
        self.layers.append(conv_layer)
        self.logger.debug("Added convolution layer")
        return self
    
    def relu(self):
        # Add ReLU activation layer
        def relu_layer(input_buffer):
            size = input_buffer.size

            # Set kernel arguments
            kernel_func = self.program.relu_activation
            kernel_func.set_arg(0, input_buffer)
            kernel_func.set_arg(1, np.int32(size))

            # Execute kernel
            global_size = (self.width, self.height)
            event = cl.enqueue_nd_range_kernel(self.queue, kernel_func, global_size, None)
            event.wait()

            self._record_time(event, "ReLU")
            return input_buffer
        
        self.layers.append(relu_layer)
        self.logger.debug("Added relu layer")
        return self
    
    def max_pool(self, pool_size):
        # Add Max pooling layer
        def max_pool_layer(input_buffer):
            mf = cl.mem_flags

            # Create buffers
            output_buffer = self._create_buffer((self.height // pool_size, self.width // pool_size))
            
            # Set kernel arguments
            kernel_func = self.program.max_pooling
            kernel_func.set_arg(0, input_buffer)
            kernel_func.set_arg(1, output_buffer)
            kernel_func.set_arg(2, np.int32(self.width))
            kernel_func.set_arg(3, np.int32(self.height))
            kernel_func.set_arg(4, np.int32(pool_size))

            # Execute kernel
            global_size = (self.width, self.height)
            event = cl.enqueue_nd_range_kernel(self.queue, kernel_func, global_size, None)
            event.wait()
            
            self._record_time(event, "Max-Pooling")
            return output_buffer
        
        self.layers.append(max_pool_layer)
        self.logger.debug("Added max pooling layer")
        return self
    
    def build(self, input_image):
        # Execute all layers
        mf = cl.mem_flags
        output = np.zeros_like(input_image)
        
        # Create buffer
        input_buffer = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_image)
        output_buffer = cl.Buffer(self.context, mf.WRITE_ONLY, input_image.nbytes)
        self.buffers.append(input_buffer)

        # Iterate over the layers
        for layer in self.layers:
            input_buffer = layer(input_buffer)
            self.buffers.append(input_buffer)

        # Get the output
        cl.enqueue_copy(self.queue, output, output_buffer)
        self.logger.debug("Succesfully copied to output image")
        
        return output
