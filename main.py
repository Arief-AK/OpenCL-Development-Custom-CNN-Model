import cv2
import numpy as np

from include.Logger import Logger
from include.Controller import Controller
from include.Visualiser import Visualiser

def LoadImage(visualiser: Visualiser) -> np.ndarray:
    image = cv2.imread("images/sample.jpg", cv2.IMREAD_GRAYSCALE)
    image_tensor = image.astype(np.float32) / 255.0 # Normalise

    visualiser.visualise_output(image_tensor, "Original-Image")
    return image_tensor

def CNNModel(controller: Controller, visualiser: Visualiser, logger:Logger):
    image = LoadImage(visualiser)
    
    controller.load_program("kernels/cnn_model.cl")
    (output, interim_results), (profiling_info) = controller.cnn_model(image)

    for layer, time in profiling_info.items():
        logger.info(f"{layer}: {time:.3f} ms")

    visualiser.visualise_output(output, "Complete Output")
    visualiser.plot_profiling(profiling_info)

if __name__ == "__main__":
    logger = Logger(__name__)
    controller = Controller()
    visualiser = Visualiser()
    controller.print_info()

    CNNModel(controller, visualiser, logger)
    logger.info("Done!")