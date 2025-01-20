#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include "yolov5.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

#if defined(RV1106_1103)
    #include "dma_alloc.hpp"
#endif

using namespace cv;
using namespace std;

const float KNOWN_DISTANCE = 5.2;
const float DOG_WIDTH = 1.6;
const float FOCAL_LENGTH = 440;

const Scalar GREEN(0, 255, 0);
const Scalar BLACK(0, 0, 0);
const string FONT = "FONT_HERSHEY_COMPLEX";


// Function to convert OpenCV Mat to image_buffer_t
int mat_to_image_buffer(const cv::Mat &frame, image_buffer_t *src_image) {
    if (frame.empty() || !src_image) {
        return -1;
    }
    
    src_image->width = frame.cols;
    src_image->height = frame.rows;
    src_image->format = IMAGE_FORMAT_RGB888;  
    src_image->size = frame.total() * frame.elemSize();
    src_image->virt_addr = (unsigned char *)malloc(src_image->size);
    
    if (!src_image->virt_addr) {
        return -1;
    }
    
    memcpy(src_image->virt_addr, frame.data, src_image->size);
    return 0;
}

// Function to ensure dimensions are aligned to 16 pixels
cv::Size get_aligned_dimensions(int width, int height) {
    int aligned_width = (width + 15) / 16 * 16;
    int aligned_height = (height + 15) / 16 * 16;
    return cv::Size(aligned_width, aligned_height);
}

float focalLengthFinder(float measured_distance, float real_width, float width_in_rf) {
        return (width_in_rf * measured_distance) / real_width;
    }

float distanceFinder(float focal_length, float real_object_width, float width_in_frame) {
        return ((real_object_width * focal_length) / width_in_frame) * 2.54 ;
    }



int main() {
    // Initialize variables
    const char *model_path = "/home/radxa/radxa/rknn_model_zoo/examples/yolov5/model/yolov5s_relu_rk3588.rknn";
    int ret;
    std::ofstream results_file("detection_results.txt");
    if (!results_file.is_open()) {
        cerr << "Error: Could not open results file!" << endl;
        return -1;
    }
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    // Initialize post-processing
    init_post_process();

    ret = init_yolov5_model(model_path, &rknn_app_ctx);
    if (ret != 0) {
        printf("init_yolov5_model fail! ret=%d model_path=%s\n", ret, model_path);
        return -1;
    }

    // Open the webcam
    // string webcam_device = "/dev/video20";

    // cout<<pipeline<<endl;
    VideoCapture cap("/dev/video20");
    
    if (!cap.isOpened()) {
        cerr << "Error: Could not open webcam!" << endl;
        return -1;
    }

    // Create window with specific size
    namedWindow("Webcam with Object Detection", WINDOW_NORMAL);
    resizeWindow("Webcam with Object Detection", 1280, 720);

    TickMeter meter;
    Mat frame, frame_rgb;
    double fps = 30.0;  // Frames per second

    while (true) {
        // Capture frame
        meter.start();
        cap >> frame;
        resize(frame, frame, Size(640, 480));
        if (frame.empty()) {
            cerr << "Error: Could not grab frame from webcam!" << endl;
            break;
        }

        // Print frame information for debugging
        cout << "Original frame info - Format: " << frame.type() 
             << ", Channels: " << frame.channels()
             << ", Size: " << frame.size() << endl;


        // Resize to aligned dimensions
        Size aligned_size = get_aligned_dimensions(frame.cols, frame.rows);
        Mat aligned_frame;
        resize(frame, aligned_frame, aligned_size);

        // Convert to RGB format
        cvtColor(aligned_frame, frame_rgb, COLOR_BGR2RGB);

        image_buffer_t src_image = {0};
        ret = mat_to_image_buffer(frame_rgb, &src_image);
        if (ret != 0) {
            cerr << "Error: Failed to convert frame to image buffer!" << endl;
            break;
        }

#if defined(RV1106_1103)
        // RV1106 specific DMA handling
        ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, src_image.size, &rknn_app_ctx.img_dma_buf.dma_buf_fd,
                           (void **)&(rknn_app_ctx.img_dma_buf.dma_buf_virt_addr));
        if (ret == 0) {
            memcpy(rknn_app_ctx.img_dma_buf.dma_buf_virt_addr, src_image.virt_addr, src_image.size);
            dma_sync_cpu_to_device(rknn_app_ctx.img_dma_buf.dma_buf_fd);
            free(src_image.virt_addr);
            src_image.virt_addr = (unsigned char *)rknn_app_ctx.img_dma_buf.dma_buf_virt_addr;
            src_image.fd = rknn_app_ctx.img_dma_buf.dma_buf_fd;
            rknn_app_ctx.img_dma_buf.size = src_image.size;
        }
#endif

        // Run inference
        object_detect_result_list od_results;
        ret = inference_yolov5_model(&rknn_app_ctx, &src_image, &od_results);
        if (ret != 0) {
            cerr << "Error: Object detection failed!" << endl;
            if (src_image.virt_addr) {
                free(src_image.virt_addr);
            }
            break;
        }

        // Draw detection results
        Mat display_frame = aligned_frame.clone();
        for (int i = 0; i < od_results.count; i++) {
            object_detect_result *det_result = &(od_results.results[i]);
            
            // Print detection results
            if (strcmp(coco_cls_to_name(det_result->cls_id), "dog") == 0){

            results_file << "Class: " << coco_cls_to_name(det_result->cls_id) 
                         << ", Box: [" << det_result->box.left << ", " 
                         << det_result->box.top << ", " 
                         << det_result->box.right << ", " 
                         << det_result->box.bottom << "], Confidence: " 
                         << det_result->prop * 100 << "%" << std::endl;

            // printf(coco_cls_to_name(det_result->cls_id));
            printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
                   det_result->box.left, det_result->box.top,
                   det_result->box.right, det_result->box.bottom,
                   det_result->prop);

            float DOG_WIDTH_REF =det_result->box.right - det_result->box.left;
            // float focal_length = focalLengthFinder(KNOWN_DISTANCE, DOG_WIDTH, DOG_WIDTH_REF);
            float distance = distanceFinder(FOCAL_LENGTH, DOG_WIDTH, DOG_WIDTH_REF);
            

            cout<< "Dog Distance:" << distance<<endl;
            cout<<"Dog Width:"<<DOG_WIDTH_REF<<endl;
            // cout<<"Focal Length"<<focal_length<<endl;

            rectangle(display_frame, 
                     Point(det_result->box.left, det_result->box.top),
                     Point(det_result->box.right, det_result->box.bottom),
                     Scalar(255, 0, 0), 2);
            
            char text[256];
            sprintf(text, "%s %.1f%%", 
                    coco_cls_to_name(det_result->cls_id), 
                    det_result->prop * 100);
            
            putText(display_frame, text, 
                    Point(det_result->box.left, det_result->box.top - 10),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
            };
        }

        
        imshow("Webcam with Object Detection", display_frame);
        meter.stop();
        fps = 1.0/meter.getTimeSec();
        meter.reset();

        cout<<"Frames Per Second:"<<fps<<endl;

        // Handle keyboard input
        char c = (char)waitKey(1);
        if (c == 27) {  // ESC key
            break;
        }

        // Cleanup
        if (src_image.virt_addr != NULL) {
#if defined(RV1106_1103)
            dma_buf_free(rknn_app_ctx.img_dma_buf.size, 
                        &rknn_app_ctx.img_dma_buf.dma_buf_fd,
                        rknn_app_ctx.img_dma_buf.dma_buf_virt_addr);
#else
            free(src_image.virt_addr);
#endif
        }
    }

    deinit_post_process();
    ret = release_yolov5_model(&rknn_app_ctx);
    if (ret != 0) {
        printf("release_yolov5_model fail! ret=%d\n", ret);
    }

    cap.release();
    destroyAllWindows();

    return 0;
}