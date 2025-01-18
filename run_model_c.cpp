#include <iostream>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

using namespace std;

void runObjectDetection(string modelPath, string imagePath) {
    string command = "/home/radxa/radxa/rknn_model_zoo/examples/yolov5/cpp/build/rknn_yolov5_demo " + modelPath + " " + imagePath;

    FILE* fp = popen(command.c_str(), "r");
    if (fp == nullptr) {
        cerr << "Error: Failed to run the command: " << command << std::endl;
        return;
    }

    char buffer[256];
    while (fgets(buffer, sizeof(buffer), fp) != nullptr) {
        cout << buffer;
    }

    fclose(fp);
}

int main() {
    string modelPath = "/home/radxa/radxa/rknn_model_zoo/examples/yolov5/model/yolov5s_relu_rk3588.rknn";
    string imagePath = "/home/radxa/radxa/rknn_model_zoo/examples/yolov5/model/bus.jpg";

    runObjectDetection(modelPath, imagePath);

    return 0;
}
