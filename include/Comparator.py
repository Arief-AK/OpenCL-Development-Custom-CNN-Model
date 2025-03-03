import numpy as np
from scipy.signal import correlate2d

from include.Logger import Logger

class Comparator:
    def __init__(self):
        self.logger = Logger(__name__)

    def convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        self.logger.debug("Performing 2D convolution")
        kernel = np.flipud(np.fliplr(kernel))
        output = correlate2d(image, kernel, mode='valid')

        return output