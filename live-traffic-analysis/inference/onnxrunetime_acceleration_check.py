#!/usr/bin/env python3

import onnxruntime


def check_onnxruntime_gpu():
    # Check ONNX Runtime version
    print("ONNX Runtime Version:", onnxruntime.__version__)

    # Check for GPU support
    try:
        if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            print("GPU Acceleration (CUDA) is available.")
        else:
            print("GPU Acceleration is not available.")
    except Exception as e:
        print("Error while checking GPU support:", e)


if __name__ == "__main__":
    check_onnxruntime_gpu()
