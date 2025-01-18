import subprocess
import os
import sys

def run_object_detection(model_path, image_path):
    if not os.path.isfile(model_path):
        print(f"Error: Model file {model_path} does not exist!")
        return
    if not os.path.isfile(image_path):
        print(f"Error: Image file {image_path} does not exist!")
        return

    command = ['/home/radxa/radxa/rknn_model_zoo/examples/yolov5/cpp/build/rknn_yolov5_demo', model_path, image_path]

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"Output:\n{result.stdout.decode()}")

    except subprocess.CalledProcessError as e:
        print(f"Error running the C program: {e}")

if __name__ == '__main__':
    model_path = r"/home/radxa/radxa/rknn_model_zoo/examples/yolov5/model/yolov5s_relu_rk3588.rknn"
    image_path = r"/home/radxa/radxa/rknn_model_zoo/examples/yolov5/model/bus.jpg"
    run_object_detection(model_path, image_path)
