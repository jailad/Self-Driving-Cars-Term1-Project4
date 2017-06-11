# Writeup for Self-Driving-Cars-Term1-Project4 - Advanced Lane Finding

[//]: # (Image References)

[image1]: ./project/output_images/detect_corners.png "Detect Corners"
[image2]: ./project/output_images/distortion_correction.png "Distortion Correction"
[image3]: ./project/output_images/thresholded_binary.png "Thresholded Binary Image"
[image4]: ./project/output_images/region_of_interest.png "Region Of Interest Selection"
[image5]: ./project/output_images/perspective_transform.png "Perspective Transform"
[image6]: ./project/output_images/histogram.png "Histogram"
[image7]: ./project/output_images/fit_lane_lines.png "Fit Lane Lines"
[image8]: ./project/output_images/generate_lane_lines.png "Generate Lane Lines"
[image10]: ./project/output_images/original_image_with_lanes.png "Original Image - with Lanes"
[image11]: ./project/output_images/original_image_with_lanes_calculations.png "Original Image - with Calculations"
[image13]: ./project/output_images/detect_lane_lines.png "Detect Lane Lines"


[video1]: ./project/project_video_output.mp4 "Project Video"
[video2]: ./project/harder_challenge_video_output_v2_2.mp4 "Harder Challenge Video"


* A submission by Jai Lad

# Table of contents

1. [Objective(s)](#objective)
2. [Key File(s)](#keyfiles)
3. [Pipeline](#pl)
    1. [Camera Calibration.](#pl1)
    2. [Distortion Correction.](#pl2)
    3. [Thresholded Binary Image.](#pl3)
    4. [Region Of Interest Selection](#pl4)
    5. [Perspective Transform.](#pl5)
    6. [Histogram for lane line detection.](#pl6)
    7. [Fit lane lines.](#pl7)
    8. [Generate and fill lane lines.](#pl8)
    9. [Original image - with lane lines.](#pl10)
    10. [Original image - with calculations.](#pl11)
    11. [Complete pipeline.](#pl12)
4. [Extended pipeline](#epl)
    1. [Rejecting spurious fitting.](#epl1)
    2. [Smoothing transitions.](#epl2)
5. [Analyzing trend of fits for project video.](#an)
6. [Future work](#fw)
    1. [Adaptive region of interest.](#fw1)
    2. [Adaptive thresholding.](#fw2)
    3. [Adaptive smoothing.](#fw4)
    4. [Using Deep Learning for lane line prediction.](#fw3)
    5. [Using weighted history.](#fw5)
    6. [Handling temporary lack of lane data.](#fw6)

<BR><BR>
---

## Objective(s) <a name="objective"></a> :


The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

<BR><BR>
---

## Key File(s) <a name="keyfiles"></a> :

* [readme.md](https://github.com/jailad/Self-Driving-Cars-Term1-Project4/blob/master/README.md) - The accompanying Readme file, with setup details.
* [project.ipynb](https://github.com/jailad/Self-Driving-Cars-Term1-Project4/blob/master/project.ipynb) - [Jupyter](http://jupyter.org/) notebook used to implement the pipeline.
* [project.py](https://github.com/jailad/Self-Driving-Cars-Term1-Project4/blob/master/project.py) - Python script version of the above notebook. ( useful for referencing specific file numbers )
* [writeup.md](https://github.com/jailad/Self-Driving-Cars-Term1-Project4/blob/master/writeup.md) - Report writeup file
* [output_images folder](https://github.com/jailad/Self-Driving-Cars-Term1-Project4/tree/master/project/output_images) - Folder with various images as generated from the pipeline.
* [pipeline video](https://youtu.be/kioah-E8Qr0) - Video of the pipeline working on the project video.
* [harder challenge video](https://youtu.be/y8AIXO57M9c) - Video of the pipeline working on the harder challenge video.
* [challenge_images](https://github.com/jailad/Self-Driving-Cars-Term1-Project4/tree/master/project/challenge_images) - A folder containing some challenging images extracted from the challenge videos.

<BR><BR>
---

## Pipeline <a name="pl"></a> :

### - Camera Calibration. <a name="pl1"></a>
* Code is linked here: [link](https://github.com/jailad/Self-Driving-Cars-Term1-Project4/blob/master/project.py#L107-L215)
* Description: Here, I used OpenCV's 'findChessboardCorners' method to detect object points and image points, followed by OpenCV 'calibrateCamera' to determine the Camera Calibration parameters.
* Image output:
* ![alt text][image1]

### - Distortion Correction.<a name="pl2"></a>
* Code is linked here: [link](https://github.com/jailad/Self-Driving-Cars-Term1-Project4/blob/master/project.py#L217-L257)
* Description: Here, I used the calibration parameters along with OpenCV's 'undistort' method to test distortion correction on an image.
* Image output: 
* ![alt text][image2]

### - Thresholded Binary Image.<a name="pl3"></a>
* Code is linked here: [link](https://github.com/jailad/Self-Driving-Cars-Term1-Project4/blob/master/project.py#L264-L322)
* Description: Here, I used a combination of Sobel-X gradient, and S-Channel thresholding to generate a thresholded binary image. I tweaked the thresholding parameter(s) until I was obtaining good results on a wide variety of images.
* Image output: 
* ![alt text][image3]

### - Region Of Interest Selection.<a name="pl4"></a>
* Code is linked here: [link](https://github.com/jailad/Self-Driving-Cars-Term1-Project4/blob/master/project.py#L326-L365)
* Description: Here, I focussed my attention to points of interest, which would be relevant from a lane detection perspective, and then rejected the other points.
* Image output: 
* ![alt text][image4]

### - Perspective Transform.<a name="pl5"></a>
* Code is linked here: [link](https://github.com/jailad/Self-Driving-Cars-Term1-Project4/blob/master/project.py#L368-L387)
* Description: Here, I converted the region of interest image, into a bird's eye view image. This was done, so that the the left and right lane lines, which actually are parallel, appeared parallel as well. This helped in fitting polynomial curves to the detected lane lines.
* Image output: 
* ![alt text][image5]

### - Histogram for lane line detection.<a name="pl6"></a>
* Code is linked here: [link](https://github.com/jailad/Self-Driving-Cars-Term1-Project4/blob/master/project.py#L400-L407)
* Description: This was done to separate the lanes into left and right lanes.
* Image output: 
* ![alt text][image6]

### - Detect and Fit lane lines.<a name="pl7"></a>
* Code is linked here: [link](https://github.com/jailad/Self-Driving-Cars-Term1-Project4/blob/master/project.py#L413-L514)
* Description: This was done to determine a good binomial fit for the detected lane lines.
* Image output(s): 
* ![alt text][image13]
* ![alt text][image7]

### - Generate and fill lane lines.<a name="pl8"></a>
* Code is linked here: [link](https://github.com/jailad/Self-Driving-Cars-Term1-Project4/blob/master/project.py#L614-L687)
* Description: Here, I used the polynomial fitted above, to synthesize continuous left and right lane lines. I then filled the area between the lane lines. I also marked the center of the lane and the center of the car so that this was easy to visualize.
* Image output(s): 
* ![alt text][image8]

### - Original image - with lane lines.<a name="pl10"></a>
* Code is linked here: [link](https://github.com/jailad/Self-Driving-Cars-Term1-Project4/blob/master/project.py#L1006)
* Description: Here, I projected back the lane line(s) onto the original image, using an inverse perspective transform.
* Image output: 
* ![alt text][image10]

### - Original image - with calculations.<a name="pl11"></a>
* Code is linked here: [link](https://github.com/jailad/Self-Driving-Cars-Term1-Project4/blob/master/project.py#L705-L774)
* Description: Here, I performed radius of curvature and vehicle location calculations.
* Image output: 
* ![alt text][image11]

### - End-to-end pipeline.<a name="pl12"></a>
* Code is linked here: [link](https://github.com/jailad/Self-Driving-Cars-Term1-Project4/blob/master/project.py#L942-L1020)
* Description: Here, I used the techniques described above, to create an end-to-end pipeline which was then tested on a few sample images, the project video, the challenge video and the harder challenge video.

<BR><BR>
---

## Extended pipeline <a name="epl"></a> :


### - Rejecting spurious fitting.<a name="epl1"></a>
* Code is linked here: [link](https://github.com/jailad/Self-Driving-Cars-Term1-Project4/blob/master/project.py#L922-L937)
* Description: Here, I kept a track of recent valid frame(s) and use a mean of these value(s) to determine if the current bottom most intersection point for the left lane and the right lane make(s) sense. If it does not make sense, then we use a mean of historical value(s). If it makes sense, then we pop the oldest entry, and update the history with this latest valid value.

### - Smoothing transitions.<a name="epl2"></a>

* Code is linked here: [link](https://github.com/jailad/Self-Driving-Cars-Term1-Project4/blob/master/project.py#L875-L892), and [link](https://github.com/jailad/Self-Driving-Cars-Term1-Project4/blob/master/project.py#L894-L920).
* Description: If we decide that the current detected fit make(s) sense, then we add it to the recent valid history, and then we generate a mean of the recent valid values. This allows the transitions to be smooth, and not 'jumpy'.

<BR><BR>
---

## Analyzing trend of fits for project video <a name="an"></a> :

* Code is linked here: [link](https://github.com/jailad/Self-Driving-Cars-Term1-Project4/blob/master/project.py#L1044-L1211).
* Description: In this section, I analyzed the various fit parameters to assess the trend of left fit coefficients and right fit coefficients as generated by the pipeline for the project video. This was used to tune the logic for validating and smoothing lane detections.

<BR><BR>
---

## Future work <a name="fw"></a> :

### - Adaptive region of interest.<a name="fw1"></a>
* Currently, I am using a hard-coded region of interest, however, for cases when we have significant curves, it is better to have an adaptive region of interest based on radius of curvature determination.

### - Adaptive thresholding.<a name="fw2"></a>
* Currently, I am using a hard-coded values for thresholded binary generation ( Sobel-X threshold, and S-Channel threshold ), however, for cases when we have a scene with less contrast, or a dark scene, it is better to have an adaptive threshold which adapts to the current scene, and which keeps a trend of the changing contrast every scene.

### - Adaptive smoothing.<a name="fw3"></a>
* Currently, I am using a fixed size of 5 valid past lanes for the left and right lane lines. This works well when the curvature(s) in the scene(s) is generally smooth. However, when the curves are very aggressive, we need to reduce the history so that we can adapt rapidly to the changing lane curvature(s).

### - Using Deep Learning for lane line prediction.<a name="fw4"></a>
* The lane detection problem could be converted to one of predicting lane line(s) given a particular scene, and then training a deep CNN on it. The input feature can be the image after region of interest selection, and the labelled output can be the 'fit line'. Infact, the pipeline for this project, can be used to quickly label a lot of images, which when combined with other tranformations like blurring, contrast changes, rotation(s), stretching etc, could be used to generate a sizable labelled data set. It would definitely be very interesting to see how this approach performs for this problem.

### - Using weighted history.<a name="fw5"></a>
* Currently, I am keeping a history of past 'X' valid left and right lane points. However, all of these are weighted equally. A better approach, would be to combine this history with a 'decay' mechanism, so that more recent value(s) are given a higher weight, relative to older valid entries.


### - Handling temporary lack of lane data<a name="fw5"></a>
* In the harder challenge video, towards the end, we temporarily have no right lane line. The pipeline would have to be adapted for this scenario because we currently assume that we should be able to threshold in a manner so as to yield a decent lane fit. 

<BR><BR>
---
