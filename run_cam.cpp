#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    string webcam_device = "/dev/video20";
    
    // Create a GStreamer pipeline with NV12 format
    // string pipeline = "v4l2src device=" + webcam_device + " ! videoconvert ! video/x-raw,format=NV12 ! mpph264enc ! appsink";
    string pipeline = "v4l2src device=" + webcam_device + " ! videoconvert ! video/x-raw,format=NV12 ! appsink";

    // Open the webcam with the GStreamer pipeline
    VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    // VideoCapture cap(webcam_device);


     double fps = cap.get(cv::CAP_PROP_FPS);

    // Check if FPS is valid (some cameras may not return FPS correctly)
    if (fps == 0.0) {
        std::cout << "FPS could not be determined (may not be supported by the camera).\n";
    } else {
        std::cout << "Camera FPS: " << fps << std::endl;
    }

    if (!cap.isOpened()) {
        cerr << "Error: Could not open webcam with GStreamer pipeline!" << endl;
        return -1;
    }

    namedWindow("Webcam", WINDOW_NORMAL);  // Create a window to display the webcam feed
    int frame_count = 0;  // Initialize frame count
    while (true) {
        Mat frame, frame_resize;
        cap >> frame;  // Grab a new frame
        resize(frame, frame_resize, Size(1920, 1080));
        // cout << "Source Image Format: " << frame<< endl;


        if (frame.empty()) {
            cerr << "Error: Could not grab frame from webcam!" << endl;
            break;
        }

        frame_count++;  // Increment the frame count
        cout << "Frame: " << frame_count << endl;

        // Check if the captured frame is in NV12 format
        if (frame.channels() == 1) {
            // If the frame is NV12, OpenCV stores it as a single channel Y-plane with an interleaved UV-plane
            cout << "Frame is in NV12 format" << endl;
            // If you want to separate the Y and UV planes, you can do that here
            Mat y_plane(frame.rows, frame.cols, CV_8UC1, frame.data);  // Y-plane
            Mat uv_plane(frame.rows / 2, frame.cols / 2, CV_8UC2, frame.data + frame.rows * frame.cols);  // UV-plane (interleaved)
            // Further processing with y_plane or uv_plane if needed
        }

        imshow("Webcam", frame_resize);  // Display the frame

        // Wait for a key press, exit if 'q' or 'Esc' is pressed
        char key = waitKey(1);
        if (key == 'q' || key == 27) { 
            break;
        }
    }

    cap.release();  // Release the VideoCapture object
    destroyAllWindows();  // Close all OpenCV windows

    return 0;
}
