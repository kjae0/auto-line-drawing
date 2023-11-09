import cv2
import numpy as np

def canny_detection(image, threshold1, threshold2, **kwargs):
    return cv2.Canny(image, 
                     threshold1=threshold1,
                     threshold2=threshold2,
                     **kwargs)
    
def gaussian_blur(image, kernel_size:tuple, sigmaX, sigmaY):
    return cv2.GaussianBlur(image, kernel_size, sigmaX, sigmaY)

def make_edge_thicker(edge, kernel_size=2):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.dilate(edge, kernel)
    
def color_merge(origin, k):
    image = gaussian_blur(origin, (15, 15), 5, 5)
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = k
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image
