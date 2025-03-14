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
        
        self.num_layers = {
            "Convolution": 0,
            "ReLU": 0,
            "Max-Pooling": 0
        }

        self.layer_buffers = {}
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

    def _store_layer_buffer_info(self, layer:str, buffer):
        # Store to layer and buffer dictionary
        current_num_layers = self.num_layers[f"{layer}"]
        self.layer_buffers.update({f"{layer}_{current_num_layers}": buffer})
        
        # Perform housekeeping
        new_num_layers = current_num_layers + 1
        self.num_layers.update({f"{layer}":new_num_layers})

    def conv2d(self, kernel, kernel_size):
        # Add convolution layer
        def conv_layer(input_buffer):
            mf = cl.mem_flags
            
            # Create buffers
            output_buffer = self._create_buffer((self.height, self.width))
            kernel_buffer = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=kernel)

            # Calculate output dimensions
            output_height = self.height - kernel_size + 1
            output_width = self.width - kernel_size + 1
            
            # Set kernel arguments
            kernel_func = self.program.convolve
            kernel_func.set_arg(0, input_buffer)
            kernel_func.set_arg(1, kernel_buffer)
            kernel_func.set_arg(2, output_buffer)
            kernel_func.set_arg(3, np.int32(self.width))
            kernel_func.set_arg(4, np.int32(self.height))
            kernel_func.set_arg(5, np.int32(kernel_size))

            # Execute kernel
            global_size = (output_width, output_height)
            event = cl.enqueue_nd_range_kernel(self.queue, kernel_func, global_size, None)
            event.wait()

            self._record_time(event, "Convolution")
            self._store_layer_buffer_info("Convolution", output_buffer)
            return output_buffer
        
        self.layers.append(conv_layer)
        self.logger.debug("Added convolution layer")
        return self
    
    def relu(self):
        # Add ReLU activation layer
        def relu_layer(input_data):
            mf = cl.mem_flags
            size = input_data.size

            # Create buffers
            output_buffer = self._create_buffer((self.height, self.width))
            input_buffer = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_data)

            # Set kernel arguments
            kernel_func = self.program.relu_activation
            kernel_func.set_arg(0, input_buffer)
            kernel_func.set_arg(1, output_buffer)
            kernel_func.set_arg(2, np.int32(size))

            # Execute kernel
            global_size = (self.width, self.height)  # Maintain 2D structure
            local_size = (min(self.BLOCK_SIZE, self.width), min(self.BLOCK_SIZE, self.height))
            event = cl.enqueue_nd_range_kernel(self.queue, kernel_func, global_size, local_size)
            event.wait()

            self._record_time(event, "ReLU")
            self._store_layer_buffer_info("ReLU", input_buffer)
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
            self._store_layer_buffer_info("Max-Pooling", output_buffer)
            return output_buffer
        
        self.layers.append(max_pool_layer)
        self.logger.debug("Added max pooling layer")
        return self
    
    def dense(self, weight_vector, bias_vector, input_size, output_size):
        def layer(input_vector):
            mf = cl.mem_flags

            # Create buffers
            input_buffer = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_vector)
            weights_buffer = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=weight_vector)
            bias_buffer = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bias_vector)
            #output_buffer = cl.Buffer(self.context, mf.WRITE_ONLY, input_vector.nbytes)
            output_buffer = self._create_buffer(input_vector.nbytes)

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

            self._record_time(event, "Dense")
            self._store_layer_buffer_info("Dense", output_buffer)
            return output_buffer
        
        self.layers.append(layer)
        self.logger.debug("Added dense layer")
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
    
    def get_tensor(self, layer, shape) -> list:
        # Initialise output list
        output_tensor_list = []
        
        # Copies OpenCL buffer to Numpy array for visualisation

        # Get the buffers from dictionary
        current_num_layers = self.num_layers[f"{layer}"]
        for index in range(current_num_layers):
            buffer = self.layer_buffers[f"{layer}_{index}"]
            output_tensor = np.zeros_like(buffer)
            cl.enqueue_copy(self.queue, output_tensor, buffer).wait()
            output_tensor_list.append(output_tensor)

        return output_tensor_list