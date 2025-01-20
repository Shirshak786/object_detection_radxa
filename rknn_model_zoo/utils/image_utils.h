#ifndef _RKNN_MODEL_ZOO_IMAGE_UTILS_H_
#define _RKNN_MODEL_ZOO_IMAGE_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "common.h"
#include "stddef.h"

/**
 * @brief LetterBox
 * 
 */
typedef struct {
    int x_pad;
    int y_pad;
    float scale;
} letterbox_t;

/**
 * @brief Read image file (support png/jpeg/bmp)
 * 
 * @param path [in] Image path
 * @param image [out] Read image
 * @return int 0: success; -1: error
 */
int read_image(const char* path, image_buffer_t* image);

/**
 * @brief Write image file (support jpg/png)
 * 
 * @param path [in] Image path
 * @param image [in] Image for write (only support IMAGE_FORMAT_RGB888)
 * @return int 0: success; -1: error
 */

 /**
 * @brief Read image from a raw byte buffer (e.g., JPEG/PNG) and store it in an image buffer.
 * 
 * @param img_data [in] Pointer to the raw image data buffer (e.g., JPEG/PNG bytes)
 * @param img_size [in] Size of the raw image data buffer
 * @param image [out] Pointer to the image_buffer_t structure where the decompressed image data will be stored
 * @return int 0: success; -1: error
 */
int read_image_from_buffer(const unsigned char* img_data, size_t img_size, image_buffer_t* image);
int write_image(const char* path, const image_buffer_t* image);

/**
 * @brief Convert image for resize and pixel format change
 * 
 * @param src_image [in] Source Image
 * @param dst_image [out] Target Image
 * @param src_box [in] Crop rectangle on source image
 * @param dst_box [in] Crop rectangle on target image
 * @param color [in] Pading color if dst_box can not fill target image
 * @return int 
 */
int convert_image(image_buffer_t* src_image, image_buffer_t* dst_image, image_rect_t* src_box, image_rect_t* dst_box, char color);

/**
 * @brief Convert image with letterbox
 * 
 * @param src_image [in] Source Image
 * @param dst_image [out] Target Image
 * @param letterbox [out] Letterbox
 * @param color [in] Fill color on target image
 * @return int 
 */
int convert_image_with_letterbox(image_buffer_t* src_image, image_buffer_t* dst_image, letterbox_t* letterbox, char color);

/**
 * @brief Get the image size
 * 
 * @param image [in] Image
 * @return int image size
 */
int get_image_size(image_buffer_t* image);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // _RKNN_MODEL_ZOO_IMAGE_UTILS_H_