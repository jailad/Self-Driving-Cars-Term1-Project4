
# coding: utf-8

# # Description

# * This file implement(s) the pipeline for the Advanced Lane Detection Project

# # Writeup

# * The project writeup is located 'here'.
# * Extra point(s) - using Deep Learning to detect lane line(s), by generating a lot of synthetic data, and then training a model to recognize the image(s)

# # Common Imports

# In[98]:

# Imports

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().magic('matplotlib inline')
import random
import pickle
from scipy import signal
import collections
import math

# Packages below needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[2]:

# Shared Constants

const_separator_line = "--------------------------------"
const_horizontal_offset = 40

const_pixel_intensity_threshold_min = 30
const_pixel_intensity_threshold_max = 100

const_plot_fontsize = 10

const_kernelsize = 9

const_src = np.float32([[120, 720], [550, 470], [700, 470], [1160, 720]])
const_dest = np.float32([[200,720], [200,0], [1080,0], [1080,720]])

const_test_straight1 = './project/test_images/straight_lines1.jpg'
const_test_straight2 = './project/test_images/straight_lines2.jpg'

const_test_image_1 = './project/test_images/test1.jpg'
const_test_image_2 = './project/test_images/test2.jpg'
const_test_image_3 = './project/test_images/test3.jpg'
const_test_image_4 = './project/test_images/test4.jpg'
const_test_image_5 = './project/test_images/test5.jpg'
const_test_image_6 = './project/test_images/test6.jpg'

const_challenge_image_paths = glob.glob('./project/challenge_images/chal*.jpg')



# In[3]:


# Useful functions to selectively turn on / off logging at different levels

const_info_log_enabled = False
def infoLog(logMessage, param_separator=None):
    if const_info_log_enabled == True:
        print("")
        if param_separator:
            print(param_separator) 
        print(logMessage)

const_debug_log_enabled = True
def debugLog(logMessage, param_separator=None):
    if const_debug_log_enabled == True:
        print("")
        if param_separator:
            print(param_separator) 
        print(logMessage)
        
const_warning_log_enabled = True
def warningLog(logMessage, param_separator=None):
    if const_warning_log_enabled == True:
        print("")
        if param_separator:
            print(param_separator) 
        print(logMessage)
        
const_error_log_enabled = True
def errorLog(logMessage, param_separator=None):
    if const_error_log_enabled == True:
        print("")
        if param_separator:
            print(param_separator) 
        print(logMessage)


# # Camera Matrix and Distortion Correction
# 
# * Determine Object points and image points for a single image

# In[28]:

# For the above loaded Random image, Detect and DrawChessboard Corners
nx = 9
ny = 6 

image_path = './project/camera_cal/calibration2.jpg'
image = mpimg.imread(image_path)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(image_gray, (nx, ny), None)

infoLog(ret)
infoLog(corners)

# If found, draw corners
if ret == True:
    # Draw and display the corners
    cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
    plt.imshow(image)


# * Determine Object points and image points for all images within the Calibration Folder

# In[5]:

# Detect Chess corners for all the images within the camera_cal folder
# This process gives us Object points and Image points, which can then be used for Camera Calibration

imagePaths = glob.glob('./project/camera_cal/calibration*.jpg')
outputPath = './project/camera_cal/output/'
infoLog(imagePaths)

nx = 9 
ny = 6 

num_channels = 3

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((ny*nx,num_channels), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2) # -1 implies shape is to be inferred
infoLog(objp)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

for index, imagePath in enumerate(imagePaths):
    img = mpimg.imread(imagePath) # Reads image as an RGB image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret == True:
        infoLog("Chessboard Corners found for - " + imagePath )
        objpoints.append(objp)
        imgpoints.append(corners)
        
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
        write_name = outputPath + 'corners_found'+str(index)+'.jpg'
        cv2.imwrite(write_name, img)

    else:
        debugLog("Chessboard Corners not found for - " + imagePath )
        

infoLog("Object Points")
infoLog(objpoints)

infoLog("Image Points")
infoLog(imgpoints)


# # Camera calibration and Example of Distortion Correction image

# In[6]:


# Test undistortion on an image
test_image_path = './project/camera_cal/calibration_test.jpg'
undistorted_test_image_output_path = './project/camera_cal/output/calibration_test_undistorted.jpg'
wide_dist_pickle_path = './project/camera_cal/output/wide_dist_pickle.p'

img = cv2.imread(test_image_path)
debugLog("Image Shape : " + str(img.shape))
img_size = (img.shape[1], img.shape[0])
debugLog("Image Size ( Note that this is Height first then Width ) : " + str(img_size))

# Camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# Save Calibration results to a Pickle file
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
dist_pickle["objpoints"] = objpoints
dist_pickle["imgpoints"] = imgpoints
pickle.dump( dist_pickle, open(wide_dist_pickle_path, "wb" ) )

# Perform Undistortion 
dst = cv2.undistort(img, mtx, dist, None, mtx)
result = cv2.imwrite(undistorted_test_image_output_path, dst)
debugLog("Successfully written to " + undistorted_test_image_output_path + " ? :-> "+ str(result) )

# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)


# In[7]:


# Read in the saved objpoints and imgpoints

dist_pickle = pickle.load( open(wide_dist_pickle_path, "rb" ) )
objpoints = dist_pickle["objpoints"]
imgpoints = dist_pickle["imgpoints"]
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

if objpoints is not None:
    debugLog("object Points saved in pickle data file restored successfully.")

# Convenience function to undistort an image, given the object points and image points
def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    img_size = (img.shape[1], img.shape[0])
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

# Convenience function to undistort an image, given the undistortion matrix
def cal_undistort_mtx(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


# # Example of Thresholded binary image

# In[8]:

# Convenience function to return a Sobel thresholded Binary Image from a given image

def abs_sobel_thresh(rgb_img, orient='x', sobel_kernel = 3, grad_thresh=(0,255)):
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY) # This assumes input image is in RGB format ( i.e. as returned by MPImg)
    sobel = None
    if orient == 'x':
        sobel =  cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
    else:
        sobel =  cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)
    abs_sobelx = np.absolute(sobel)
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    binary_output = np.zeros_like(scaled_sobelx)
    binary_output[(scaled_sobelx >= grad_thresh[0]) & (scaled_sobelx <= grad_thresh[1])] = 1

    return binary_output

# Convenience function to apply S Channel Thresholding on an image
def apply_s_channel_thresholding(rgb_img, thresh=(0,255)):
    # Convert to HLS
#     img_hls = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HLS).astype(np.float)
    img_hls = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HLS)
    # Apply threshold to S channel
    S = img_hls[:,:,2]
    binary_s = np.zeros_like(S)
    binary_s[(S > thresh[0]) & (S <= thresh[1])] = 1

    return binary_s

def thresholding(img, binary_threshold=(20, 100), s_channel_threshold=(170,255), sobel_kernel_size=const_kernelsize):
    binary_thresholded_img_x = abs_sobel_thresh(img, 'x', sobel_kernel_size, binary_threshold)
    s_channel_thresholded_image = apply_s_channel_thresholding(img, s_channel_threshold)
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(binary_thresholded_img_x)
    combined_binary[(binary_thresholded_img_x == 1) | (s_channel_thresholded_image == 1)] = 1
    return combined_binary

# Define Thresholding parameters 
local_sobel_kernel = 1
local_sobel_threshold = (30,100)
local_s_channel_threshold = (170,255)

# Perform Thresholding
img = mpimg.imread(const_test_straight1) # Reads image as an RGB image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Force to RGB format
infoLog(img.shape)
sobel_x_thresholded_image = abs_sobel_thresh(img, 'x', local_sobel_kernel, local_sobel_threshold)
s_channel_thresholded_image = apply_s_channel_thresholding(img, local_s_channel_threshold)
combined_image = thresholding(img)

# Visualize Thresholding
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=10)
ax2.imshow(sobel_x_thresholded_image, cmap='gray')
ax2.set_title('Sobel-X Thresholded Image', fontsize=10)
ax3.imshow(s_channel_thresholded_image, cmap='gray')
ax3.set_title('S-Channel Thresholded Image', fontsize=10)
ax4.imshow(combined_image, cmap='gray')
ax4.set_title('Combined Thresholded Image', fontsize=10)



# # Region of Interest Selection

# In[9]:


# Convenience function for region of interest selection

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    region_of_interest_image = cv2.bitwise_and(img, mask)
    return region_of_interest_image


image_shape = combined_image.shape
vertices = np.array([[(0,image_shape[0]),(550, 470), (750, 470), (image_shape[1],image_shape[0])]], dtype=np.int32)

# Perform RoI Selection
region_of_interest_image = region_of_interest(combined_image, vertices)

# Visualize RoI Selection
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,10))
ax1.imshow(combined_image, cmap='gray')
ax1.set_title('Combined Thresholded Image', fontsize=10)
ax2.imshow(region_of_interest_image, cmap='gray')
ax2.set_title('Region of Interest image', fontsize=10)



# # Example of Perspective Transform

# In[10]:


# Convenience function to apply a perspective transform to an image, given certain Source and Destination parameters
def warp_src_dest(img, src, dest):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dest)
    Minv = cv2.getPerspectiveTransform(dest, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, M, Minv

warped_image, M, Minv = warp_src_dest(region_of_interest_image, const_src, const_dest)

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,10))
ax1.imshow(combined_image, cmap='gray')
ax1.set_title('Region Of Interest Image', fontsize=10)
ax2.imshow(warped_image, cmap='gray')
ax2.set_title('Perspective Transformed image', fontsize=10)



# # Lane Line Pixel determination, and curve fitting

# * Determine left lane and right lane with Histogram analysis

# In[11]:


def collapse_into_flat_arrays(leftx, lefty, rightx, righty):
    leftx = [x
             for array in leftx
             for x in array]
    lefty = [x
             for array in lefty
             for x in array]
    rightx = [x
              for array in rightx
              for x in array]
    righty = [x
              for array in righty
              for x in array]

    leftx = np.array(leftx)
    lefty = np.array(lefty)
    rightx = np.array(rightx)
    righty = np.array(righty)

    return leftx, lefty, rightx, righty

def get_pixel_in_window(img, x_center, y_center, size):
    """
    returns selected pixels inside a window.
    :param img: binary image
    :param x_center: x coordinate of the window center
    :param y_center: y coordinate of the window center
    :param size: size of the window in pixel
    :return: x, y of detected pixels
    """
    half_size = size // 2
    window = img[int(y_center - half_size):int(y_center + half_size), int(x_center - half_size):int(x_center + half_size)]

    x, y = (window.T == 1).nonzero()

    x = x + x_center - half_size
    y = y + y_center - half_size

    return x, y

def histogram_pixels(warped_thresholded_image, offset=50, steps=6,
                     window_radius=200, medianfilt_kernel_size=51,
                     horizontal_offset=50, enable_visualization = False):
        
    # Initialise arrays
    left_x = []
    left_y = []
    right_x = []
    right_y = []

    # Parameters
    height = warped_thresholded_image.shape[0]
    offset_height = height - offset
    width = warped_thresholded_image.shape[1]
    half_frame = warped_thresholded_image.shape[1] // 2
    pixels_per_step = offset_height / steps

    for step in range(steps):
        left_x_window_centres = []
        right_x_window_centres = []
        y_window_centres = []

        # Define the window (horizontal slice)
        window_start_y = height - (step * pixels_per_step) + offset
        window_end_y = window_start_y - pixels_per_step + offset

        # Take a count of all the pixels at each x-value in the horizontal slice
        histogram = np.sum(warped_thresholded_image[int(window_end_y):int(window_start_y), int(horizontal_offset):int(width - horizontal_offset)], axis=0)
        
        if(enable_visualization == True):
            plt.plot(histogram)

        # Smoothen the histogram
        histogram_smooth = signal.medfilt(histogram, medianfilt_kernel_size)

        if(enable_visualization == True):
            plt.plot(histogram_smooth)

        # Identify the left and right peaks
        left_peaks = np.array(signal.find_peaks_cwt(histogram_smooth[:half_frame], np.arange(1, 10)))
        right_peaks = np.array(signal.find_peaks_cwt(histogram_smooth[half_frame:], np.arange(1, 10)))
        if len(left_peaks) > 0:
            left_peak = max(left_peaks)
            left_x_window_centres.append(left_peak)

        if len(right_peaks) > 0:
            right_peak = max(right_peaks) + half_frame
            right_x_window_centres.append(right_peak)

        # Add coordinates to window centres

        if len(left_peaks) > 0 or len(right_peaks) > 0:
            y_window_centres.append((window_start_y + window_end_y) // 2)

        # Get pixels in the left window
        for left_x_centre, y_centre in zip(left_x_window_centres, y_window_centres):
            left_x_additional, left_y_additional = get_pixel_in_window(warped_thresholded_image, left_x_centre,
            
                                                                       y_centre, window_radius)
            if(enable_visualization == True):
                plt.scatter(left_x_additional, left_y_additional)
            # Add pixels to list
            left_x.append(left_x_additional)
            left_y.append(left_y_additional)

        # Get pixels in the right window
        for right_x_centre, y_centre in zip(right_x_window_centres, y_window_centres):
            right_x_additional, right_y_additional = get_pixel_in_window(warped_thresholded_image, right_x_centre,
                                                                         y_centre, window_radius)
            if(enable_visualization == True):
                plt.scatter(right_x_additional, right_y_additional)
            # Add pixels to list
            right_x.append(right_x_additional)
            right_y.append(right_y_additional)
                
    if len(right_x) == 0 or len(left_x) == 0:
        print("X co ordinate for left or right peak not found")

        horizontal_offset = 0

        left_x = []
        left_y = []
        right_x = []
        right_y = []

        for step in range(steps):
            left_x_window_centres = []
            right_x_window_centres = []
            y_window_centres = []

            # Define the window (horizontal slice)
            window_start_y = height - (step * pixels_per_step) + offset
            window_end_y = window_start_y - pixels_per_step + offset

            # Take a count of all the pixels at each x-value in the horizontal slice
            histogram = np.sum(warped_thresholded_image[int(window_end_y):int(window_start_y),
                               int(horizontal_offset):int(width - horizontal_offset)], axis=0)
            
            if(enable_visualization == True):
                plt.plot(histogram)

            # Smoothen the histogram
            histogram_smooth = signal.medfilt(histogram, medianfilt_kernel_size)

            if(enable_visualization == True):
                plt.plot(histogram_smooth)

            # Identify the left and right peaks
            left_peaks = np.array(signal.find_peaks_cwt(histogram_smooth[:half_frame], np.arange(1, 10)))
            right_peaks = np.array(signal.find_peaks_cwt(histogram_smooth[half_frame:], np.arange(1, 10)))
            if len(left_peaks) > 0:
                left_peak = max(left_peaks)
                left_x_window_centres.append(left_peak)

            if len(right_peaks) > 0:
                right_peak = max(right_peaks) + half_frame
                right_x_window_centres.append(right_peak)

            # Add coordinates to window centres

            if len(left_peaks) > 0 or len(right_peaks) > 0:
                y_window_centres.append((window_start_y + window_end_y) // 2)

            # Get pixels in the left window
            for left_x_centre, y_centre in zip(left_x_window_centres, y_window_centres):
                left_x_additional, left_y_additional = get_pixel_in_window(warped_thresholded_image, left_x_centre,
                                                                           y_centre, window_radius)
                if(enable_visualization == True):
                    plt.scatter(left_x_additional, left_y_additional)
                # Add pixels to list
                left_x.append(left_x_additional)
                left_y.append(left_y_additional)

            # Get pixels in the right window
            for right_x_centre, y_centre in zip(right_x_window_centres, y_window_centres):
                right_x_additional, right_y_additional = get_pixel_in_window(warped_thresholded_image, right_x_centre,
                                                                             y_centre, window_radius)
                if(enable_visualization == True):
                    plt.scatter(right_x_additional, right_y_additional)
                # Add pixels to list
                right_x.append(right_x_additional)
                right_y.append(right_y_additional)
                        
    return collapse_into_flat_arrays(left_x, left_y, right_x, right_y)


leftx, lefty, rightx, righty = histogram_pixels(warped_image, horizontal_offset=const_horizontal_offset, enable_visualization = True)

debugLog(leftx)
debugLog(lefty)
debugLog(rightx)
debugLog(righty)




# * Fit a polynomial to the above points 

# In[12]:


def fit_second_order_poly(indep, dep, return_coeffs=False):
    fit = np.polyfit(indep, dep, 2)
    fitdep = fit[0]*indep**2 + fit[1]*indep + fit[2]
    if return_coeffs == True:
        return fitdep, fit
    else:
        return fitdep, None

def get_fit_equations(lefty,leftx, righty, rightx, return_coeffs=False):
    left_fit, left_coeffs = fit_second_order_poly(lefty, leftx, return_coeffs=return_coeffs)
    right_fit, right_coeffs = fit_second_order_poly(righty, rightx, return_coeffs=return_coeffs)
    return left_fit, left_coeffs, right_fit, right_coeffs

left_fit, left_coeffs, right_fit, right_coeffs = get_fit_equations(lefty,leftx, righty, rightx, True)

plt.plot(left_fit, lefty, color='green', linewidth=3)
plt.plot(right_fit, righty, color='green', linewidth=3)
plt.imshow(warped_image, cmap="gray")
lanes = plt.gcf()

debugLog("Fit and coefficients: ")
debugLog(left_fit.shape)
debugLog(left_coeffs.shape)
debugLog(right_fit.shape)
debugLog(right_coeffs.shape)

debugLog("X, Y co ordinates for left and right lane(s): ")
debugLog(lefty)
debugLog(leftx)
debugLog(righty)
debugLog(rightx)




# * Generate a solid line which represents the lane line(s), based on the fit above

# In[13]:


def lane_poly(yval, poly_coeffs):
    """Returns x value for poly given a y-value.
    Note here x = Ay^2 + By + C."""
    return poly_coeffs[0]*yval**2 + poly_coeffs[1]*yval + poly_coeffs[2]

def draw_poly(img, poly, poly_coeffs, steps, color=[255, 0, 0], thickness=10, dashed=False):
    img_height = img.shape[0]
    pixels_per_step = img_height // steps

    for i in range(steps):
        start = i * pixels_per_step
        end = start + pixels_per_step

        start_point = (int(poly(start, poly_coeffs=poly_coeffs)), start)
        end_point = (int(poly(end, poly_coeffs=poly_coeffs)), end)

        if dashed == False or i % 2 == 1:
            img = cv2.line(img, end_point, start_point, color, thickness)

    return img

def get_generated_lane_lines(param_left_coeffs, param_right_coeffs):
    blank_canvas = np.zeros((720, 1280))
    polyfit_left = draw_poly(blank_canvas, lane_poly, param_left_coeffs, 30)
    polyfit_drawn = draw_poly(polyfit_left, lane_poly, param_right_coeffs, 30)
    return polyfit_drawn
    
polyfit_drawn = get_generated_lane_lines(left_coeffs, right_coeffs)

plt.imshow(polyfit_drawn, cmap="gray")


# * Create a 'filled' area which can be projected back to the original image

# In[14]:


def evaluate_poly(indep, poly_coeffs):
    return poly_coeffs[0]*indep**2 + poly_coeffs[1]*indep + poly_coeffs[2]

def highlight_lane_line_area(mask_template, left_poly, right_poly, start_y = 0, end_y = 720):
    area_mask = mask_template
    for y in range(start_y, end_y):
        left = evaluate_poly(y, left_poly)
        right = evaluate_poly(y, right_poly)
        area_mask[y][int(left):int(right)] = 1

    return area_mask

def get_filled_lanes(param_polyfit_drawn, param_left_coeffs, param_right_coeffs):
    blank_canvas = np.zeros((720, 1280))
    colour_canvas = cv2.cvtColor(blank_canvas.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    filled_lanes = colour_canvas
    filled_lanes[param_polyfit_drawn > 1] = [0,0,255]
    area = highlight_lane_line_area(blank_canvas, param_left_coeffs, param_right_coeffs)
    filled_lanes[area == 1] = [0,255,0]
    return filled_lanes

filled_lanes = get_filled_lanes(polyfit_drawn, left_coeffs, right_coeffs)

plt.imshow(filled_lanes)



# * Project the highlighted area back to the original image

# In[15]:


def project_lanes_on_original(param_img, param_filled_lanes, param_inverse_perspective_transform_matrix):
    image_shape = param_img.shape
    lane_lines = cv2.warpPerspective(param_filled_lanes, param_inverse_perspective_transform_matrix, (image_shape[1], image_shape[0]), flags=cv2.INTER_LINEAR)
    combined_img = cv2.add(lane_lines, param_img)
    return combined_img

lane_lines_detected_image = project_lanes_on_original(img, filled_lanes, Minv)

plt.imshow(lane_lines_detected_image)




# # Radius of Curvature, and Vehicle Position Calculation

# * Do the RoC Calculation

# In[16]:


def center(y, left_poly, right_poly):
    center = (1.5 * evaluate_poly(y, left_poly)
              - evaluate_poly(y, right_poly)) / 2
    return center


def perform_calculations(param_left_coeffs, param_right_coeffs):
    y_eval = 500
    left_curverad = np.absolute(((1 + (2 * param_left_coeffs[0] * y_eval + param_left_coeffs[1])**2) ** 1.5)                 /(2 * left_coeffs[0]))
    right_curverad = np.absolute(((1 + (2 * param_right_coeffs[0] * y_eval + param_right_coeffs[1]) ** 2) ** 1.5)                  /(2 * right_coeffs[0]))
    infoLog("Left lane curve radius: " + str(left_curverad) + "pixels")
    infoLog("Right lane curve radius: "+ str(right_curverad) + "pixels")
    curvature = (left_curverad + right_curverad) / 2
    centre = center(719, left_coeffs, right_coeffs)
    min_curvature = min(left_curverad, right_curverad)
    return curvature, centre, min_curvature
    
curvature, centre, min_curvature = perform_calculations(left_coeffs, right_coeffs)
    


# # Final image, projected back to the original image

# * Add the RoC Calculation to the image

# In[17]:

def add_calculations_to_image(img, curvature, vehicle_position, min_curvature, left_coeffs=(0,0,0), right_coeffs=(0,0,0)):
    """
    Draws information about the center offset and the current lane curvature onto the given image.
    :param img:
    """
    return_img = img.copy()
    
    # Convert from pixels to meters
    vehicle_position = vehicle_position / 12800 * 3.7
    curvature = curvature / 128 * 3.7
    min_curvature = min_curvature / 128 * 3.7

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(return_img, 'Radius of Curvature = %d(m)' % curvature, (50, 50), font, 1, (255, 255, 255), 2)
    left_or_right = "left" if vehicle_position < 0 else "right"
    cv2.putText(return_img, 'Vehicle is %.2fm %s of center' % (np.abs(vehicle_position), left_or_right), (50, 100), font, 1,
                (255, 255, 255), 2)
    cv2.putText(return_img, 'Min Radius of Curvature = %d(m)' % min_curvature, (50, 150), font, 1, (255, 255, 255), 2)
    cv2.putText(return_img, 'Left poly coeffs = %.3f %.3f %.3f' % (left_coeffs[0], left_coeffs[1], left_coeffs[2]), (50, 200), font, 1, (255, 255, 255), 2)
    cv2.putText(return_img, 'Right poly coeffs = %.3f %.3f %.3f' % (right_coeffs[0], right_coeffs[1], right_coeffs[2]), (50, 250), font, 1, (255, 255, 255), 2)
    return return_img

return_img = add_calculations_to_image(lane_lines_detected_image, curvature=curvature, 
                     vehicle_position=centre, 
                     min_curvature=min_curvature,
                     left_coeffs=left_coeffs,
                     right_coeffs=right_coeffs)

plt.imshow(return_img)


# # Pipeline Code

# In[123]:


# The number of valid history objects to store for validation, and interpolation
const_history_size = 5

# The maximum allowed difference between the X co ordinate of the previously stored lane bottom intersection points ( left, and right ) 
# as compared to the current estimated bottom intersection point
const_max_lane_bottom_intersection_x_valid_threshold = 300

global_left_coeffs_history = collections.deque(maxlen=const_history_size)
global_right_coeffs_history = collections.deque(maxlen=const_history_size)

global_left_lane_bottom_intersection_x = collections.deque(maxlen=const_history_size)
global_right_lane_bottom_intersection_x = collections.deque(maxlen=const_history_size)

def reset_history():
    global global_left_coeffs_history
    global global_right_coeffs_history
    global global_left_lane_bottom_intersection_x
    global global_right_lane_bottom_intersection_x
    global_left_coeffs_history = collections.deque(maxlen=const_history_size)
    global_right_coeffs_history = collections.deque(maxlen=const_history_size)
    global_left_lane_bottom_intersection_x = collections.deque(maxlen=const_history_size)
    global_right_lane_bottom_intersection_x = collections.deque(maxlen=const_history_size)

def update_history_coeffs(left_coeff, right_coeff):
    global global_left_coeffs_history
    global global_right_coeffs_history
    
    # Add the items to History
    if(left_coeff != None):
        global_left_coeffs_history.appendleft(left_coeff)
        
    if(right_coeff != None):
        global_right_coeffs_history.appendleft(right_coeff)
    
    infoLog(global_left_coeffs_history)
    infoLog(global_right_coeffs_history)
    
def get_coeffs_from_history():
    global global_left_coeffs_history
    global global_right_coeffs_history
    
    # Add the items to History
    left_coeffs_list = list(global_left_coeffs_history)
    right_coeffs_list = list(global_right_coeffs_history)
    
    infoLog(global_left_coeffs_history)
    infoLog(global_right_coeffs_history)
    
    left_coeff_history = np.mean(left_coeffs_list, axis=0)
    right_coeff_history = np.mean(right_coeffs_list, axis=0)
    
    infoLog(left_coeff_history)
    infoLog(right_coeff_history)
    
    return left_coeff_history, right_coeff_history


def update_history_bottom_intersection_x(left_bottom_intersection_x, right_bottom_intersection_x):
    global global_left_lane_bottom_intersection_x
    global global_right_lane_bottom_intersection_x
    
    # Add the items to History
    if(left_bottom_intersection_x != None):
        global_left_lane_bottom_intersection_x.appendleft(left_bottom_intersection_x)
    
    if(right_bottom_intersection_x != None):
        global_right_lane_bottom_intersection_x.appendleft(right_bottom_intersection_x)
    
    infoLog(global_left_lane_bottom_intersection_x)
    infoLog(global_right_lane_bottom_intersection_x)

def get_bottom_intersection_points_x_from_history():
    global global_left_lane_bottom_intersection_x
    global global_right_lane_bottom_intersection_x
    
    left_bottom_intersection_point_x_history = np.mean(global_left_lane_bottom_intersection_x)
    right_bottom_intersection_point_x_history = np.mean(global_right_lane_bottom_intersection_x)
    
    debugLog(global_left_lane_bottom_intersection_x)
    debugLog(global_right_lane_bottom_intersection_x)
    
    return left_bottom_intersection_point_x_history, right_bottom_intersection_point_x_history

    
def is_lane_bottom_x_intersection_valid(current_intersection_point_x, history_intersection_point_x):
    if(math.isnan(history_intersection_point_x) == False):
        difference = abs(current_intersection_point_x-history_intersection_point_x)
        if (difference <= const_max_lane_bottom_intersection_x_valid_threshold):
            infoLog("Current point is valid.")
            return True
        else:
            debugLog("Current intersection point appears to be invalid ( relative to history ).")
            debugLog("Current point : " + str(current_intersection_point_x))
            debugLog("Historical point ( mean ) : " + str(history_intersection_point_x))
            return False
    else:
        debugLog("Returning valid, because currently we do not have any history.")
        return True

def pipeline(imageOrPath, isImagePath = False, enable_visualization = False, write_to_disk = False):
    # Reference a few necessary global variables
    global global_left_coeffs
    global global_right_coeffs

    # If the image is a path, then read the image from Path
    image = None
    if(isImagePath == True):
        image = mpimg.imread(imageOrPath) # Reads image as an RGB image
    else:
        image = imageOrPath

    # Clear out matplotlib plot frame for each run
    plt.clf()

    # Distortion correction
    dc_image = cal_undistort_mtx(image, mtx, dist)
    
    # Thresholding
    thresholded_image = thresholding(dc_image,(40, 100),(180,255))

    # RoI selection
    image_shape = image.shape
    vertices = np.array([[(0,image_shape[0]),(550, 470), (750, 470), (image_shape[1],image_shape[0])]], dtype=np.int32)
    region_of_interest_image = region_of_interest(thresholded_image, vertices)

    # Perspective transform
    warped_image, M, Minv = warp_src_dest(region_of_interest_image, const_src, const_dest)
    
    # Lane detection using transform
    leftx, lefty, rightx, righty = histogram_pixels(warped_image, horizontal_offset=const_horizontal_offset)

    # Fit a polynomial to the detected lanes
    try:
        
        # Attempt a fit ( may lead to exception if the fit fails )
        left_fit, left_coeffs, right_fit, right_coeffs = get_fit_equations(lefty, leftx, righty, rightx, True)
        
        # If we get a fit then we need to validate the fit relative to recent history
        left_bottom_intersection_point_x_history, right_bottom_intersection_point_x_history = get_bottom_intersection_points_x_from_history()
        
        # Validate left lane fit
        left_lane_lowest_point = lane_poly(720, left_coeffs)
        if(is_lane_bottom_x_intersection_valid(left_lane_lowest_point, left_bottom_intersection_point_x_history)):
            update_history_bottom_intersection_x(left_lane_lowest_point, None)
            update_history_coeffs(left_coeffs, None)

        # Validate right lane fit
        right_lane_lowest_point = lane_poly(720, right_coeffs)
        if(is_lane_bottom_x_intersection_valid(right_lane_lowest_point, right_bottom_intersection_point_x_history)):
            update_history_bottom_intersection_x(None, right_lane_lowest_point)
            update_history_coeffs(None, right_coeffs)
        
        left_coeffs, right_coeffs = get_coeffs_from_history()
    
    except TypeError:
        
        errorLog("TypeError occured, unable to fit a line so using coefficients from history.")
        left_coeffs, right_coeffs = get_coeffs_from_history() 
    
    # Lane generation
    polyfit_drawn = get_generated_lane_lines(left_coeffs, right_coeffs)
    
    # Fill lane area
    filled_lanes = get_filled_lanes(polyfit_drawn, left_coeffs, right_coeffs)
    
    # Project area back to original image
    lane_lines_detected_image = project_lanes_on_original(dc_image, filled_lanes, Minv)

    # Perform Radius of curvature calculations
    curvature, centre, min_curvature = perform_calculations(left_coeffs, right_coeffs)

    # Add these calculation(s) to the original image
    return_img = add_calculations_to_image(lane_lines_detected_image, curvature=curvature, 
                     vehicle_position=centre, 
                     min_curvature=min_curvature,
                     left_coeffs=left_coeffs,
                     right_coeffs=right_coeffs)
       
    # Visualize the pipeline, if needed
    if(enable_visualization == True):
        
        f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(10, 1, figsize=(60,80))
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=10)
        ax2.imshow(dc_image)
        ax2.set_title('Distortion Corrected Image', fontsize=10)
        ax3.imshow(thresholded_image, cmap='gray')
        ax3.set_title('Thresholded Image', fontsize=10)
        ax4.imshow(region_of_interest_image, cmap='gray')
        ax4.set_title('Region Of Interest Image', fontsize=10)
        ax5.imshow(warped_image, cmap='gray')
        ax5.set_title('Perspective transformed', fontsize=10)
        ax6.plot(left_fit, lefty, color='green', linewidth=3)
        ax6.plot(right_fit, righty, color='green', linewidth=3)
        ax6.imshow(warped_image, cmap="gray")
        ax7.imshow(polyfit_drawn, cmap='gray')
        ax7.set_title('Generated lane lines', fontsize=10)
        ax8.imshow(filled_lanes, cmap='gray')
        ax8.set_title('Filled lane lines', fontsize=10)
        ax9.imshow(lane_lines_detected_image, cmap='gray')
        ax9.set_title('Lane line(s) on original image', fontsize=10)
        ax10.imshow(return_img, cmap='gray')
        ax10.set_title('Final image with Calculations', fontsize=10)
        if(write_to_disk == True): 
            plt.savefig("./project/output_images/output.png", bbox_inches='tight')
    
    # Return the image
    return return_img

reset_history()
combined_img = pipeline(const_test_straight1, True, True, True)


# # Pipeline as executed on test Images

# In[19]:


reset_history()
combined_img = pipeline(const_test_image_2, True, True)


# In[20]:


reset_history()
combined_img = pipeline(const_test_image_3, True, True)


# In[21]:


reset_history()
combined_img = pipeline(const_test_image_4, True, True)



# In[22]:


reset_history()
combined_img = pipeline(const_test_image_5, True, True)



# In[23]:


reset_history()
combined_img = pipeline(const_test_image_6, True, True)



# # Pipeline as executed on the Project Video

# In[ ]:


reset_history()

clip1 = VideoFileClip("./project/project_video.mp4")
output_clip1 = clip1.fl_image(pipeline)
output1 = './project/project_video_output3.mp4'
get_ipython().magic('time output_clip1.write_videofile(output1, audio=False)')


# # Pipeline as executed on a few Challenge Image(s)

# In[ ]:


for imagePath in const_challenge_image_paths:
    img = mpimg.imread(imagePath) # Reads image as an RGB image
    infoLog(imagePath)
    reset_history()



# # Pipeline as executed on a few Challenge Video(s)

# In[115]:


reset_history()

clip2 = VideoFileClip("./project/challenge_video.mp4")
output_clip2 = clip2.fl_image(pipeline)
output2 = './project/challenge_video_output3.mp4'
get_ipython().magic('time output_clip2.write_videofile(output2, audio=False)')


# In[ ]:


reset_history()

clip3 = VideoFileClip("./project/harder_challenge_video.mp4")
output_clip3 = clip3.fl_image(pipeline)
output3 = './project/harder_challenge_video_output3.mp4'
get_ipython().magic('time output_clip3.write_videofile(output3, audio=False)')


# 
# # TBD :
# 
# * Improve smoothing logic.

# In[ ]:



