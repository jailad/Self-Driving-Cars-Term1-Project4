
# coding: utf-8

# # Description

# * This notebook implement(s) the pipeline for the Advanced Lane Detection Project.

# # Writeup

# * The project writeup is located at : https://github.com/jailad/Self-Driving-Cars-Term1-Project4/blob/master/writeup.md
# 

# # Imports

# In[249]:

# Imports

import numpy as np
import cv2
import glob # Used to read in image files of a particular pattern
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().magic('matplotlib inline')
import random
import pickle
import collections # Used to store a recent window of good fits
import math # Used for nan detection

# Packages below needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# # Constants

# In[250]:

# Shared Constants

const_separator_line = "--------------------------------"

const_measurements_fontsize = 1
const_measurements_fontcolor = (255,255,255)

const_kernelsize = 9 # Larger Kernel size -> 'Smoother' detection

# Constants for perspective transform
const_src = np.float32([[120, 720], [550, 470], [700, 470], [1160, 720]])
const_dest = np.float32([[200,720], [200,0], [1080,0], [1080,720]])

# Constants representing paths of a few test images
const_test_straight1 = './project/test_images/straight_lines1.jpg'
const_test_straight2 = './project/test_images/straight_lines2.jpg'

const_test_image_1 = './project/test_images/test1.jpg'
const_test_image_2 = './project/test_images/test2.jpg'
const_test_image_3 = './project/test_images/test3.jpg'
const_test_image_4 = './project/test_images/test4.jpg'
const_test_image_5 = './project/test_images/test5.jpg'
const_test_image_6 = './project/test_images/test6.jpg'

const_challenge_image_paths = glob.glob('./project/challenge_images/chal*.jpg')



# # Useful Utility method(s)

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


# # Preparing for Camera Calibration and Distortion Correction
# 
# * Determine Object points and image points for a single image

# In[287]:

# Detect and DrawChessboard Corners

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

# In[288]:

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

# In[289]:


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


# # Ensuring that the Distortion Correction parameters can be restored from Disk

# In[290]:


# Read in the saved objpoints and imgpoints
dist_pickle = pickle.load( open(wide_dist_pickle_path, "rb" ) )
objpoints = dist_pickle["objpoints"]
imgpoints = dist_pickle["imgpoints"]
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

if mtx is not None:
    debugLog("Distortion correction matrix restored successfully from Disk.")

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

# In[291]:

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

# Convenience function to combine the above thresholding techniques into a single method
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

# In[303]:


# Convenience function for region of interest selection
# This code is the same as my submission from Project 1 for the lane detection project.

def region_of_interest(img, vertices):

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

# In[293]:


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



# # Using Histogram technique for :
# * Left lane detection.
# * Right lane detection.
# * Generating a Binomial fit for left lane.
# * Generating a Binomial fit for right lane.

# In[294]:


# Generating the Histogram

def generate_histogram(warped_image):
    histogram = np.sum(warped_image[warped_image.shape[0]//2:,:], axis=0)
    plt.plot(histogram)

generate_histogram(warped_image)
plt.savefig('./project/output_images/histogram.png', bbox_inches='tight')


# In[304]:


# Use Histogram for the botom half of the image to separate lane line(s) into left lane and right lane pixels 
# using Windowing technique.
# Once left and right lane pixels have been separated, fit a second order polynomial to the detected left and right lane(s)
# Once we have a 'fit' we generate synthetic lane line(s) using the fit and plot those on the image in yellow color.

def detect_lanes_and_fit_poly(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    infoLog(leftx.shape)

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
#     plt.plot(left_fitx, ploty, color='yellow')
#     plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
#     plt.savefig('./project/output_images/detect_lane_lines.png', bbox_inches='tight')

    infoLog(left_fit)
    infoLog(right_fit)
    
    # Calculate and return the bottom most lane intersection points
    y_bottom = 719
    leftx_bottom = int(left_fit[0]*y_bottom**2 + left_fit[1]*y_bottom + left_fit[2])
    rightx_bottom = int(right_fit[0]*y_bottom**2 + right_fit[1]*y_bottom + right_fit[2])

    return left_fit, right_fit, left_fitx, right_fitx, ploty, left_lane_inds, right_lane_inds, leftx_bottom, rightx_bottom

    
left_fit, right_fit, left_fitx, right_fitx, ploty, left_lane_inds, right_lane_inds, leftx_bottom, rightx_bottom  = detect_lanes_and_fit_poly(warped_image)


# In[296]:



# Code to visualize the selection window
def visualize_selection_window(binary_warped, left_lane_inds, right_lane_inds):
    margin = 100
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(result)
#     plt.plot(left_fitx, ploty, color='yellow')
#     plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

visualize_selection_window(warped_image, left_lane_inds, right_lane_inds)


# In[305]:


# Generate some fake data to represent lane-line pixels
ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
quadratic_coeff = 3e-4 # arbitrary quadratic coefficient
# For each y position generate random x position within +/-50 pix
# of the line base position in each case (x=200 for left, and x=900 for right)
leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-100, high=101) 
                              for y in ploty])
rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-100, high=101) 
                                for y in ploty])

infoLog(leftx.shape)

leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
rightx = rightx[::-1]  # Reverse to match top-to-bottom in y



# Fit a second order polynomial to pixel positions in each fake lane line
left_fit = np.polyfit(ploty, leftx, 2)
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fit = np.polyfit(ploty, rightx, 2)
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

# Plot up the fake data
mark_size = 3
plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
plt.xlim(0, 1280)
plt.ylim(0, 720)
plt.plot(left_fitx, ploty, color='green', linewidth=3)
plt.plot(right_fitx, ploty, color='green', linewidth=3)
plt.gca().invert_yaxis() # to visualize as we do the images

plt.savefig('./project/output_images/fit_lane_lines.png', bbox_inches='tight')



# In[306]:



# Define y-value where we want radius of curvature
# I'll choose the maximum y-value, corresponding to the bottom of the image
y_eval = np.max(ploty)
left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
infoLog(str(left_curverad) + ',' + str(right_curverad))
# Example values: 1926.74 1908.48



# In[302]:


# Once lane(s) have been detected, we will the region between them
# We also draw a line representing lane center
# We also draw a small ( red ) line representing center of the car

def fill_lane(warped, original_image, left_fit, right_fit):
    
    (image_height, image_width) = warped.shape
    car_center_x = int(image_width/2)
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Obtain lane points corresponding to the fit
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    infoLog(np.min(left_fitx))
    infoLog(np.max(left_fitx))
    infoLog(np.min(right_fitx))
    infoLog(np.max(right_fitx))
    
    y_bottom = 719
    y_top = 0
    
    leftx_bottom = int(left_fit[0]*y_bottom**2 + left_fit[1]*y_bottom + left_fit[2])
    leftx_top = int(left_fit[0]*y_top**2 + left_fit[1]*y_top + left_fit[2])
    rightx_bottom = int(right_fit[0]*y_bottom**2 + right_fit[1]*y_bottom + right_fit[2])
    rightx_top = int(right_fit[0]*y_top**2 + right_fit[1]*y_top + right_fit[2])   
    
    centerx_bottom = int((leftx_bottom + rightx_bottom)/2)
    centerx_top = int((leftx_top + rightx_top)/2)
        
    # Calculations to get the bottom pixel
    left_bottom = (leftx_bottom, y_bottom)
    left_top = (leftx_top, y_top)
    right_bottom = (rightx_bottom, y_bottom)
    right_top = (rightx_top, y_top)
    
    center_bottom = (centerx_bottom, y_bottom)
    center_top = (centerx_top, y_top)
    
    car_center_bottom = (car_center_x, y_bottom)
    car_center_top = (car_center_x, y_bottom - 5)

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Draw lane edges
    cv2.line(color_warp, left_bottom, left_top, (255,255, 0), 50)
    cv2.line(color_warp, right_bottom, right_top, (255,255, 0), 50)
    
    # Draw lane center
    cv2.line(color_warp, center_bottom, center_top, (255,255, 0), 10, 1)
    
    # Draw car center
    cv2.line(color_warp, car_center_bottom, car_center_top, (255,0, 0), 10, 1)
        
#     plt.imshow(color_warp)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(original_image, 1, newwarp, 0.5, 0)
    return result


img = fill_lane(warped_image, img, left_fit, right_fit)
# plt.imshow(img)
# cv2.imwrite('./project/output_images/fill_lane_lines.png', img)


# 
# # Get Measurements like:
# * Left Radius of curvature.
# * Right Radius of curvature.
# * Mean Radius of curvature. 
# * Vehicle position relative to lane center.
# * 'Detected' X-coordinate of bottom of left lane.
# * 'Detected' X-coordinate of bottom of right lane.
# 

# In[237]:


# Get relevant measurements, based on the fit values
def get_measurements(left_fit, right_fit):
    ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    mean_curverad = (left_curverad + right_curverad) / 2.0

    # Now our radius of curvature is in meters
    infoLog(str(left_curverad) + 'm' + str(right_curverad) + 'm' + str(mean_curverad) + 'm')
    
    # Perform Distance from center calculation
    car_center = 640
    y_bottom = 719
    
    leftx_bottom = int(left_fit[0]*y_bottom**2 + left_fit[1]*y_bottom + left_fit[2])
    rightx_bottom = int(right_fit[0]*y_bottom**2 + right_fit[1]*y_bottom + right_fit[2])
    lane_center_bottom = (leftx_bottom + rightx_bottom) / 2
    
    relative_position = car_center - lane_center_bottom
    relative_position_meters = relative_position*xm_per_pix
    relative_position_string = "right of lane center" if relative_position_meters > 0 else "left of lane center"
    vehicle_position_string = "Car is " + str(np.abs(round(relative_position_meters, 4))) + " m " + relative_position_string 
    
    return left_curverad, right_curverad, mean_curverad, vehicle_position_string, relative_position_meters, leftx_bottom, rightx_bottom


left_curverad, right_curverad, mean_curverad, vehicle_position_string, relative_position_meters, leftx_bottom, rightx_bottom  = get_measurements(left_fit, right_fit)

debugLog(vehicle_position_string)


# 
# # Add the above measurements to the original image
# 
# 

# In[248]:


# Add the different relevant measurements to an input image
def add_measurements_to_image(param_img, left_curverad, right_curverad, mean_curverad, vehicle_position_string, history_frames, valid_lane_threshold):
    return_img = param_img.copy()

    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(return_img, 'Radius of Curvature - Left = %d(m)' % left_curverad, (50, 50), font, const_measurements_fontsize, const_measurements_fontcolor, 2)
    cv2.putText(return_img, 'Radius of Curvature - Right = %d(m)' % right_curverad, (50, 100), font, const_measurements_fontsize, const_measurements_fontcolor, 2)
    cv2.putText(return_img, 'Radius of Curvature - Mean = %d(m)' % mean_curverad, (50, 150), font, const_measurements_fontsize, const_measurements_fontcolor, 2)
    cv2.putText(return_img, '%s' % vehicle_position_string, (50, 200), font, const_measurements_fontsize, const_measurements_fontcolor, 2)
    cv2.putText(return_img, 'Number of smoothing frames = %d' % history_frames, (900, 50), font, const_measurements_fontsize, const_measurements_fontcolor, 2)
    cv2.putText(return_img, 'Valid Lane Detection Threshold = %d(px)' % valid_lane_threshold, (900, 100), font, const_measurements_fontsize, const_measurements_fontcolor, 2)

#     cv2.imwrite("calculations.jpg", return_img)

    return return_img

return_img = add_measurements_to_image(img, 1,2,1.5,3 , 20, 100 )

plt.imshow(return_img)


# # Lane ( History ) Management Code
# 
# * Code for doing the following:
#     * Keeping a track of all left and right 'fits'.
#     * Keeping a track of past 'X' valid left and right fits.
#     * Validating a fit.
#     * Smoothing a fit. 
#     * Resetting the history for a new run.

# In[300]:


# The number of valid history objects to store for validation, and interpolation
const_history_size = 50
const_max_lane_bottom_intersection_x_valid_threshold = 70

# List(s) below were used to view the general trend of various calculations for the pipeline
global_left_fit_list = list()
global_right_fit_list = list()
global_left_curverad_list = list()
global_right_curverad_list = list()
global_leftx_bottom_list = list()
global_rightx_bottom_list = list()

# List(s) below was used to : 1. Store last 5 history points 2. Bottom 'X' intersection points for left lane and right lane, 3. Evaluate if the detected new intersection points made sense.
global_left_coeffs_history = collections.deque(maxlen=const_history_size)
global_right_coeffs_history = collections.deque(maxlen=const_history_size)
global_left_lane_bottom_intersection_x = collections.deque(maxlen=const_history_size)
global_right_lane_bottom_intersection_x = collections.deque(maxlen=const_history_size)

# Reset all state
def reset_history():
    global global_left_fit_list
    global global_right_fit_list
    global global_left_curverad_list
    global global_right_curverad_list
    global global_leftx_bottom_list
    global global_rightx_bottom_list
    
    global global_left_coeffs_history
    global global_right_coeffs_history
    global global_left_lane_bottom_intersection_x
    global global_right_lane_bottom_intersection_x
    
    global_left_fit_list = None
    global_right_fit_list = None
    global_left_curverad_list = None
    global_right_curverad_list = None
    global_leftx_bottom_list = None
    global_rightx_bottom_list = None
    
    global_left_fit_list = list()
    global_right_fit_list = list()
    global_left_curverad_list = list()
    global_right_curverad_list = list()
    global_leftx_bottom_list = list()
    global_rightx_bottom_list = list()
    
    global_left_coeffs_history = collections.deque(maxlen=const_history_size)
    global_right_coeffs_history = collections.deque(maxlen=const_history_size)
    global_left_lane_bottom_intersection_x = collections.deque(maxlen=const_history_size)
    global_right_lane_bottom_intersection_x = collections.deque(maxlen=const_history_size)

# Update state values with currently detected values
def update_history(param_left_fit, param_right_fit, param_left_curverad, param_right_curverad, param_leftx_bottom, param_rightx_bottom):
    global global_left_fit_list
    global global_right_fit_list
    global global_left_curverad_list
    global global_right_curverad_list
    global global_leftx_bottom_list
    global global_rightx_bottom_list
    
    global_left_fit_list.append(param_left_fit)
    global_right_fit_list.append(param_right_fit) 
    global_left_curverad_list.append(param_left_curverad)
    global_right_curverad_list.append(param_right_curverad)
    global_leftx_bottom_list.append(param_leftx_bottom)
    global_rightx_bottom_list.append(param_rightx_bottom)
    
    update_history_bottom_intersection_x(param_leftx_bottom, param_rightx_bottom)

# Update the left fit history
def update_history_left_fit(param_left_fit):
    global global_left_coeffs_history
    
    # Add the items to History
    if(param_left_fit != None):
        global_left_coeffs_history.appendleft(param_left_fit)

# Update the right fit history
def update_history_right_fit(param_right_fit):
    global global_right_coeffs_history
    
    # Add the items to History
    if(param_right_fit != None):
        global_right_coeffs_history.appendleft(param_right_fit)

# Get the average value of the past valid fit coefficients from history
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

# Update the historical record of the left lane bottom intersection point and right lane bottom intersection point
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

# Fetch the average value of the recently detected intersection points for the left lane and the right lane
def get_bottom_intersection_points_x_from_history():
    global global_left_lane_bottom_intersection_x
    global global_right_lane_bottom_intersection_x
    
    left_bottom_intersection_point_x_history = np.mean(global_left_lane_bottom_intersection_x)
    right_bottom_intersection_point_x_history = np.mean(global_right_lane_bottom_intersection_x)
    
    infoLog(global_left_lane_bottom_intersection_x)
    infoLog(global_right_lane_bottom_intersection_x)
    
    return left_bottom_intersection_point_x_history, right_bottom_intersection_point_x_history

# Signals of a good fit:
# * bottom_left_x, and bottom_right_x are close to the average of the past few X co ordinate values
def is_lane_bottom_x_intersection_valid(current_intersection_point_x, history_intersection_point_x):
    if(math.isnan(history_intersection_point_x) == False):
        difference = abs(current_intersection_point_x-history_intersection_point_x)
        if (difference <= const_max_lane_bottom_intersection_x_valid_threshold):
            infoLog("Current point is valid.")
            return True
        else:
            debugLog("Current point : " + str(current_intersection_point_x))
            debugLog("Historical point ( mean ) : " + str(history_intersection_point_x))
            debugLog("Current point is invalid.")
            return False
    else:
        debugLog("Returning valid, because currently we do not have any history.")
        return True




# # Pipeline
# 
# * Define the Pipeline.
# * Execute it on a test image as a quick check.

# In[307]:


def pipeline_v2(imageOrPath, isImagePath = False, enable_visualization = False, write_to_disk = False):

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
    
    # Perform lane fit
    left_fit, right_fit, left_fitx, right_fitx, ploty, left_lane_inds, right_lane_inds, leftx_bottom, rightx_bottom = detect_lanes_and_fit_poly(warped_image)    
    
    # Evaluate Quality of fit
    ## Get historical data
    left_bottom_intersection_point_x_history, right_bottom_intersection_point_x_history = get_bottom_intersection_points_x_from_history()
    
    # Evaluate left fit 
    left_fit_valid = is_lane_bottom_x_intersection_valid(leftx_bottom, left_bottom_intersection_point_x_history)
    
    # If left fit is valid
    ## Update Left coefficient 'Window' history with this value
    if left_fit_valid == True:
        update_history_left_fit(left_fit)

    # Evaluate right fit 
    right_fit_valid = is_lane_bottom_x_intersection_valid(rightx_bottom, right_bottom_intersection_point_x_history)
    
    # If right fit is valid
    ## Update Left coefficient 'Window' history with this value
    if right_fit_valid == True:
        update_history_right_fit(right_fit)
    
    ## Fetch the 'valid' and 'smoothened' fit from history
    left_fit, right_fit = get_coeffs_from_history()
    
    # Perform necessary calculations
    left_curverad, right_curverad, mean_curverad, vehicle_position_string, relative_position_meters, leftx_bottom, rightx_bottom = get_measurements(left_fit, right_fit)
    
    # Overlay filled lane line(s) onto original image
    return_img = fill_lane(warped_image, dc_image, left_fit, right_fit)

    update_history(left_fit, right_fit, left_curverad, right_curverad, leftx_bottom, rightx_bottom)

    # Add the calculations to the image
    return_img = add_measurements_to_image(return_img, left_curverad, right_curverad, mean_curverad, vehicle_position_string, const_history_size, const_max_lane_bottom_intersection_x_valid_threshold )
           
    # Return the image
    return return_img

reset_history()
combined_img = pipeline_v2(const_test_image_4, True, True, True)
plt.imshow(combined_img)

cv2.imwrite('./project/output_images/original_image_with_lanes_calculations.png', combined_img)


# # Execute the Pipeline on the Project Video

# In[242]:


reset_history()
clip1 = VideoFileClip("./project/project_video.mp4")
output_clip1 = clip1.fl_image(pipeline_v2)
output1 = './project/project_video_output.mp4'
get_ipython().magic('time output_clip1.write_videofile(output1, audio=False)')


# # Analyze the fits generated for project video
# * In the section below we :
#     * Plot the various fit coefficients.
#     * Plot the radius of curvature values.
# * We do this to get a better understanding of :
#     * How the pipeline sees the road.
#     * When the pipeline fails, what does it see.
# * The results of this analysis were then used retroactively in refining the pipeline for smoothing lane detections.

# ## Analyze Fits

# In[217]:


# Analyze fits
debugLog(len(global_left_fit_list))
debugLog(len(global_right_fit_list))
debugLog(len(global_left_curverad_list))
debugLog(len(global_right_curverad_list))
debugLog(len(global_leftx_bottom_list))
debugLog(len(global_rightx_bottom_list))

left_fit_coeff_0 = [coeffs[0] for coeffs in global_left_fit_list]
left_fit_coeff_1 = [coeffs[1] for coeffs in global_left_fit_list]
left_fit_coeff_2 = [coeffs[2] for coeffs in global_left_fit_list]

right_fit_coeff_0 = [coeffs[0] for coeffs in global_right_fit_list]
right_fit_coeff_1 = [coeffs[1] for coeffs in global_right_fit_list]
right_fit_coeff_2 = [coeffs[2] for coeffs in global_right_fit_list]


# ## Analyze Left Lane Coefficient #0

# In[218]:


debugLog(np.mean(left_fit_coeff_0))
debugLog(np.std(left_fit_coeff_0))
debugLog(2*np.std(left_fit_coeff_0))

plt.plot(left_fit_coeff_0)
# plt.plot(rolling_left_fit_coeff_0)
plt.ylabel('Left Coefficients - [0]')
plt.show()


# ## Analyze Left Lane Coefficient #1

# In[219]:


debugLog(np.mean(left_fit_coeff_1))

plt.plot(left_fit_coeff_1)
plt.ylabel('Left Coefficients - [1]')
plt.show()



# ## Analyze Left Lane Coefficient #2

# In[220]:


plt.plot(left_fit_coeff_2)
plt.ylabel('Left Coefficients - [2]')
plt.show()


# ## Analyze Right Lane Coefficient #0

# In[221]:


plt.plot(right_fit_coeff_0)
plt.ylabel('Right Coefficients - [0]')
plt.show()


# ## Analyze Right Lane Coefficient #1

# In[222]:


plt.plot(right_fit_coeff_1)
plt.ylabel('Right Coefficients - [1]')
plt.show()


# ## Analyze Right Lane Coefficient #2

# In[223]:


plt.plot(right_fit_coeff_2)
plt.ylabel('Right Coefficients - [2]')
plt.show()


# ## Analyze Left Radius of curvature

# In[224]:


# Left Radius of Curvature

# Descriptive Statistics
debugLog(np.mean(global_left_curverad_list))
debugLog(np.min(global_left_curverad_list))
debugLog(np.max(global_left_curverad_list))
debugLog(np.std(global_left_curverad_list))
debugLog(2*np.std(global_left_curverad_list))

# Plot
plt.plot(global_left_curverad_list)
plt.ylabel('Left Radius Of Curvature')
plt.show()


# ## Analyze Right Radius of curvature

# In[225]:


# Right Radius of Curvature

# Descriptive Statistics
debugLog(np.mean(global_right_curverad_list))
debugLog(np.min(global_right_curverad_list))
debugLog(np.max(global_right_curverad_list))
debugLog(np.std(global_right_curverad_list))
debugLog(2*np.std(global_right_curverad_list))

# Plot
plt.plot(global_right_curverad_list)
plt.ylabel('Right Radius Of Curvature')
plt.show()


# ## Analyze Left Lane Bottom - Detected intersection point

# In[226]:


# Left bottom detected lane intersection point

# Descriptive Statistics
debugLog(np.mean(global_leftx_bottom_list))
debugLog(np.min(global_leftx_bottom_list))
debugLog(np.max(global_leftx_bottom_list))
debugLog(np.std(global_leftx_bottom_list))
debugLog(2*np.std(global_leftx_bottom_list))

# Plot
plt.plot(global_leftx_bottom_list)
plt.ylabel('Left Lane - Bottom - predicted X Co ordinate')
plt.show()


# ## Analyze Right Lane Bottom - Detected intersection point

# In[227]:


# Right bottom detected lane intersection point

# Descriptive Statistics
debugLog(np.mean(global_rightx_bottom_list))
debugLog(np.min(global_rightx_bottom_list))
debugLog(np.max(global_rightx_bottom_list))
debugLog(np.std(global_rightx_bottom_list))
debugLog(2*np.std(global_rightx_bottom_list))

# Plot
plt.plot(global_rightx_bottom_list)
plt.ylabel('Right Lane - Bottom - predicted X Co ordinate')
plt.show()


# # Pipeline as executed on a few other test Images

# In[228]:


reset_history()
combined_img = pipeline_v2(const_test_image_2, True, True, True)
plt.imshow(combined_img)


# In[230]:


reset_history()
combined_img = pipeline_v2(const_test_image_3, True, True, True)
plt.imshow(combined_img)


# In[231]:


reset_history()
combined_img = pipeline_v2(const_test_image_4, True, True, True)
plt.imshow(combined_img)


# In[232]:


reset_history()
combined_img = pipeline_v2(const_test_image_5, True, True, True)
plt.imshow(combined_img)


# In[233]:


reset_history()
combined_img = pipeline_v2(const_test_image_6, True, True, True)
plt.imshow(combined_img)


# # Pipeline as executed on a few Challenge Image(s)
# 
# * From the challenge video(s), I picked a few challenging video frame(s), and ran the pipeline on these images.

# In[253]:


reset_history()
combined_img = pipeline_v2(const_challenge_image_paths[0], True, True, True)
plt.imshow(combined_img)


# In[254]:


reset_history()
combined_img = pipeline_v2(const_challenge_image_paths[1], True, True, True)
plt.imshow(combined_img)


# In[255]:


reset_history()
combined_img = pipeline_v2(const_challenge_image_paths[2], True, True, True)
plt.imshow(combined_img)


# In[256]:


reset_history()
combined_img = pipeline_v2(const_challenge_image_paths[3], True, True, True)
plt.imshow(combined_img)


# In[257]:


reset_history()
combined_img = pipeline_v2(const_challenge_image_paths[4], True, True, True)
plt.imshow(combined_img)


# In[258]:


reset_history()
combined_img = pipeline_v2(const_challenge_image_paths[5], True, True, True)
plt.imshow(combined_img)


# In[259]:


reset_history()
combined_img = pipeline_v2(const_challenge_image_paths[6], True, True, True)
plt.imshow(combined_img)


# In[260]:


reset_history()
combined_img = pipeline_v2(const_challenge_image_paths[7], True, True, True)
plt.imshow(combined_img)


# In[261]:


reset_history()
combined_img = pipeline_v2(const_challenge_image_paths[8], True, True, True)
plt.imshow(combined_img)


# In[262]:


reset_history()
combined_img = pipeline_v2(const_challenge_image_paths[9], True, True, True)
plt.imshow(combined_img)


# In[263]:


reset_history()
combined_img = pipeline_v2(const_challenge_image_paths[10], True, True, True)
plt.imshow(combined_img)


# In[264]:


reset_history()
combined_img = pipeline_v2(const_challenge_image_paths[11], True, True, True)
plt.imshow(combined_img)


# In[265]:


reset_history()
combined_img = pipeline_v2(const_challenge_image_paths[12], True, True, True)
plt.imshow(combined_img)


# In[266]:


reset_history()
combined_img = pipeline_v2(const_challenge_image_paths[13], True, True, True)
plt.imshow(combined_img)


# In[267]:


reset_history()
combined_img = pipeline_v2(const_challenge_image_paths[14], True, True, True)
plt.imshow(combined_img)


# In[268]:


reset_history()
combined_img = pipeline_v2(const_challenge_image_paths[15], True, True, True)
plt.imshow(combined_img)


# # Pipeline as executed on the Challenge Video
# 
# * Currently, this is leading to a TypeError because a fit cannot be found for a frame.
# * The resolution strategy for this scenario, would be to pick a different set of parameters for thresholding, so that the lane(s) can be detected, and fits can be generated.

# In[240]:


reset_history()

clip2 = VideoFileClip("./project/challenge_video.mp4")
output_clip2 = clip2.fl_image(pipeline_v2)
output2 = './project/challenge_video_output_v2_2.mp4'
get_ipython().magic('time output_clip2.write_videofile(output2, audio=False)')


# # Pipeline as executed on the Harder Challenge Video
# 
# * This runs surprisingly well, and produced a decent enough result without any specific tweaks for this video.
# * It fails at the very end because of the contrast situation and the steepness of the curve.

# In[270]:


reset_history()

clip3 = VideoFileClip("./project/harder_challenge_video.mp4")
output_clip3 = clip3.fl_image(pipeline_v2)
output3 = './project/harder_challenge_video_output_v2_2.mp4'
get_ipython().magic('time output_clip3.write_videofile(output3, audio=False)')


# 
# # Future Work :
# 
# * Currently, when the pipeline is executed on a video, after a valid lane detection, I can narrow down the region to search for a valid lane. I can implement this to improve detection quality, and also reduce lane detection time.
# 
# * Tweak pipeline so that it produces good results for all the Challenge photos.
# 
# * Tweak pipeline so that it produces good results for both Challenge videos.
# 
# * Tweak pipeline so that it produces good results for a Self captured video.
# 
# * Explore other thresholding techniques ( other than sobel x and s channel thresholding ) for more challenging images, such that they enable the program to run properly on challenging photos and videos.
# 
# * If we do not get a lane close to the region of interest, we need 'go back' to our thresholding approach and modify it accordingly to detect the lane lines. Try to find lane line(s) with this new strategy a couple of times.
# 
# * For valid radius of curvature(s), we need to predict how quickly the RoC is changing, so that we can modify the size of the smoothing window accordingly. This could be accomplished by taking a derivative of the RoC and understanding it to set a baseline, and then seeing if the current derivative is higher or lower than this. If the derivative is higher, we need to reduce the size of the smoothing window. Else, we can increase the smoothing window ( upto a threshold )
# 
# * If we write to disk at each stage of the pipeline, we should be able to produce a compelling visualization showing how the pipeline detected lane lines at different segments of the video.
# 

# In[ ]:



