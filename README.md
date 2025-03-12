# OpenCL Development: Custom CNN Model

Application of OpenCL for machine learning model development. This work demonstrates the development of layers for machine learning models. The outcome of this project is to produce a custom CNN machine learning model

## OpenCL setup
OpenCL is typically packaged with graphic drivers from vendors like **AMD**, **Intel**, and **NVIDIA**. To ensure that OpenCL is properly installed on your system, install the latest graphic drivers on your device.

- For AMD GPUs, download drivers from the [AMD website](https://www.amd.com/en/resources/support-articles/faqs/GPU-56.html).
- For NVIDIA GPUs, download drivers from the [NVIDIA website](https://www.nvidia.com/en-us/drivers/).
- For Intel GPUs, download drivers from the [Intel website](https://www.intel.com/content/www/us/en/download-center/home.html).

### Linux
On Linux machines, it is recommended to install the `ocl-icd-opencl-dev` package
```shell
sudo apt-get install ocl-icd-opencl-dev clinfo
```

## Getting Started
> [!Important]\
> This project utilises Python's Virtual Environment module. It is recommended to install dependencies within a virtual environment.

To create and initialise a Python virtual environment, run the following commands.
```shell
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

To run the benchmarking of layer models, run the following command.
```shell
python Benchmark.py
```