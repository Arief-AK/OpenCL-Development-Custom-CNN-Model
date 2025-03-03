import time
import numpy as np
from scipy.signal import correlate2d

from include.Logger import Logger

class Comparator:
    def __init__(self):
        self.logger = Logger(__name__)

    def convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> tuple:
        self.logger.debug("Performing 2D convolution")
        kernel = np.flipud(np.fliplr(kernel))

        start_time = time.time()
        output = correlate2d(image, kernel, mode='valid')
        end_time = time.time()

        elapsed_time = (end_time - start_time) * 1000
        return output, elapsed_time