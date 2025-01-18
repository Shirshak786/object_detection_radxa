#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "yolov5.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

#if defined(RV1106_1103)
    #include "dma_alloc.hpp"
#endif

using namespace cv;
using namespace std;

// Function to convert OpenCV Mat to image_buffer_t
int mat_to_image_buffer(const cv::Mat &frame, image_buffer_t *src_image) {
    // Allocate buffer and copy data
    src_image->size = frame.total() * frame.elemSize();
    src_image->virt_addr = (unsigned char *)malloc(src_image->size);
    if (!src_image->virt_addr) {
        return -1;
    }
    memcpy(src_image->virt_addr, frame.data, src_image->size);
    return 0;
}

int main() {
    // Initialize variables
    const char *model_path = "/home/radxa/radxa/rknn_model_zoo/examples/yolov5/model/yolov5s_relu_rk3588.rknn";  // Replace with your model path
    int ret;
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
    VideoCapture cap("/dev/video20");  // Use default camera (adjust for the correct device path)
    if (!cap.isOpened()) {
        cerr << "Error: Could not open webcam!" << endl;
        return -1;
    }
    namedWindow("Webcam with Object Detection", WINDOW_NORMAL);  // Create window for webcam feed

    while (true) {
        Mat frame;
        cap >> frame;  // Capture frame from webcam
        if (frame.empty()) {
            cerr << "Error: Could not grab frame from webcam!" << endl;
            break;
        }

        // Convert the frame to image_buffer_t
        image_buffer_t src_image;
        ret = mat_to_image_buffer(frame, &src_image);
        if (ret != 0) {
            std::cerr << "Error: Failed to convert frame to image buffer!" << std::endl;
            break;
        }

#if defined(RV1106_1103)
        // If running on RV1106, handle DMA buffer allocation for input/output
        ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, src_image.size, &rknn_app_ctx.img_dma_buf.dma_buf_fd,
                            (void **)&(rknn_app_ctx.img_dma_buf.dma_buf_virt_addr));
        memcpy(rknn_app_ctx.img_dma_buf.dma_buf_virt_addr, src_image.virt_addr, src_image.size);
        dma_sync_cpu_to_device(rknn_app_ctx.img_dma_buf.dma_buf_fd);
        free(src_image.virt_addr);
        src_image.virt_addr = (unsigned char *)rknn_app_ctx.img_dma_buf.dma_buf_virt_addr;
        src_image.fd = rknn_app_ctx.img_dma_buf.dma_buf_fd;
        rknn_app_ctx.img_dma_buf.size = src_image.size;
#endif

        // Run object detection
        object_detect_result_list od_results;
        ret = inference_yolov5_model(&rknn_app_ctx, &src_image, &od_results);
        if (ret != 0) {
            std::cerr << "Error: Object detection failed!" << std::endl;
            break;
        }

        // Draw bounding boxes and labels
        char text[256];
        for (int i = 0; i < od_results.count; i++) {
            object_detect_result *det_result = &(od_results.results[i]);
            printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
                   det_result->box.left, det_result->box.top,
                   det_result->box.right, det_result->box.bottom,
                   det_result->prop);
            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;

            draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);
            sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
            draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
        }

        // Convert processed buffer back to cv::Mat for displaying
        frame = cv::Mat(frame.size(), CV_8UC3, src_image.virt_addr);

        // Show the frame with bounding boxes
        imshow("Webcam with Object Detection", frame);

        // Exit the loop if the user presses the 'Esc' key
        char c = (char)waitKey(1);
        if (c == 27) {  // ASCII value for 'Esc'
            break;
        }

        // Free the memory allocated for image buffer
        if (src_image.virt_addr != NULL) {
#if defined(RV1106_1103)
            dma_buf_free(rknn_app_ctx.img_dma_buf.size, &rknn_app_ctx.img_dma_buf.dma_buf_fd,
                         rknn_app_ctx.img_dma_buf.dma_buf_virt_addr);
#else
            free(src_image.virt_addr);
#endif
        }
    }

    // Release the model and cleanup
    deinit_post_process();
    ret = release_yolov5_model(&rknn_app_ctx);
    if (ret != 0) {
        printf("release_yolov5_model fail! ret=%d\n", ret);
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
