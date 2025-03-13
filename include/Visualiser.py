import matplotlib.pyplot as plt

from include.Logger import Logger

class Visualiser:
    def __init__(self):
        self.logger = Logger(__name__)

    def visualise_output(self, output, title: str):
        # Visualise 2D tensor as an image
        plt.figure(figsize=(6, 6))
        plt.imshow(output, cmap='gray')
        plt.colorbar()
        plt.title(title)
        plt.axis("off")
        plt.savefig(f"images/{title}.png")

    def plot_profiling(self, profiling_info: dict):
        # Plot execution times
        layers = list(profiling_info.keys())
        times = list(profiling_info.values())

        plt.figure(figsize=(10, 5))
        plt.barh(layers, times, color='skyblue')
        plt.xlabel("Execution time (ms)")
        plt.title("CNN Layer Profiling")
        plt.savefig("images/layer_timings.png")