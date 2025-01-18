#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    VideoCapture cap("/dev/video20");  //Lower USB address /dev/video20

    if (!cap.isOpened()) {
        cerr << "Error: Could not open webcam!" << endl;
        return -1;
    }
    namedWindow("Webcam", WINDOW_NORMAL);

    while (true) {
        Mat frame;

        cap >> frame;
        if (frame.empty()) {
            cerr << "Error: Could not grab frame from webcam!" << endl;
            break;
        }
        imshow("Webcam", frame);

        char key = waitKey(1);
        if (key == 'q' || key == 27) { 
            break;
        }
    }
    cap.release();
    destroyAllWindows();

    return 0;
}
