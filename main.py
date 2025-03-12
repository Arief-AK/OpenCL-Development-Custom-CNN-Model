import numpy as np

from include.Logger import Logger
from include.Controller import Controller

IMAGE_SIZE = 1024

def RunModel(controller: Controller, logger:Logger):
    image = np.random.rand(IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)   # Random image
    
    controller.load_program("kernels/cnn_model.cl")
    output, profiling_info = controller.cnn_model(image)

    for layer, time in profiling_info.items():
        logger.info(f"{layer}: {time:.3f} ms")

if __name__ == "__main__":
    logger = Logger(__name__)
    controller = Controller()
    controller.print_info()

    RunModel(controller, logger)