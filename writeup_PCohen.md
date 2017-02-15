##Writeup 
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration1_undistorted.png "Undistorted view of calibration image 1"
[image2]: ./output_images/exampleFrame_uncorrected.png "Distorted example frame"
[image3]: ./output_images/exampleFrame_distortionCorrected.png "Undistorted example frame"
[image4]: ./output_images/exampleFrame_thresholded.png "Thresholded frame"
[image5]: ./output_images/exampleFrame_topView.png "Transformed (Top-View) image"
[image6]: ./output_images/exampleFrame_CentroidBoxes.png "Centroid Boxes"
[image7]: ./output_images/exampleFrame_CentroidBoxesExcluded.png "Centroid Boxes Excluded"
[image8]: ./output_images/exampleFrame_PolynomialsDrawn.png "Plynomials drawn"
[image9]: ./output_images/exampleFrame_LaneSuperImposed.png "Lane area superimposed"
[image10]: ./output_images/exampleFrame_annotated.png "Geometric information"

[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The camera calibration is performed in the custom-built *"*calibrateCamera*"* function (starting at line 10 of helperFunctions.py). 

To calculate the the camera matrix and the distotion coefficients I utilized the calibrateCamera function of the openCV python library. Among other things, this function expects an array of image points and an array of corresponding object points as inputs. The image points were obtained by using OpenCVs findChessboardCorners function with the 20 provided calibration images. The object points are constructed by appending the expected 2D-coordinates of the corners of an ideal, flat chessboard to an initially empty vector 20 times (once for each calibration image).

Finally, the images are corrected (undistorted) by using openCVs *unsdistort* function (line 64 of advancedLaneFind.py) 

![alt text][image1]
Example of distorion-corrected calbration image

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.

![alt text][image2]
Example of a uncorrected image taken from the video


![alt text][image3]
Example of a distortion-corrected image taken from the video


####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
The thresholding is done via a combination of different gradient thresholds as well as color thresholds. The combination is defined in function *"*thresholding*"* as defined in helperFunctions.py line 96 and following). This function makes use of the following thresholding techniques:
* Absolute sobel gradient thresholding (in x-direction) (defined in helperFunctions.py line 43 and following)
* Magnitude sobel gradient thresholding (defined in helperFunctions.py line 60 and following)
* Thresholding by direction of sobel gradient (defined in helperFunctions.py line 73 and following)
* Thresholding by Hue (color channel thresholding defined in helperFunctions.py line 86 and following))
* Thresholding by Saturation


To enable the thresholding function, the image had to be converted to grayscale as well as to HLS mode with the hel of the OpenCV function cvtColor

In the main script advancedLaneFind.py, the thresholding is called on line 70

![alt text][image4]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transform is called in advancedLaneFind.py on line 76 and defined in helperfunctions.py on line 133. It accepts the thresholded image and ultimately performs the transformation via the openCV function warpPerspective. The transformation matrix used in warpPerspective is created via openCV's getPerspectiveTransform function. 

getPerspectiveTransform requires a set of source pints as well as destination points to calculate the transformation matrix. These points were defined manually and hardcoded as follows:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 211, 718.85   | 361, 718.85   | 
| 603.97, 445   | 361, 0        |
| 676.54, 445   | 950, 0        |
| 1100, 718.85  | 950, 718.85   |


I was able to verify that the perspective transform was working properly by applying it to the provided straight-line test images and verifying that the lane lines appeared close to parallel in the transformed image.

![alt text][image5]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The first big step I take is to condense the numerous pixels that are deemed to belong to the lane markers down to a set of "centroids" for each lane line (that is one set for left lane line, one set for right lane line).

Depending on the situation I use two slightly different methods for calculating the positions of the centroids.

**Method 1**

Method 1 is called *find_window_centroids* and is defined starting on line 259 of helperfunctions.py. 
This method divides the transformed (top view) image into several (currently 9) horizontal slices (called levels). For each slice, it finds two "centroids" i.e. the center positions of the pixels presumably belonging to the left and the right lane markers, respectively. 
The vertical position of those centroids is simply the mid-point between the bottom and the top of the respective horizontal slice.
The lateral position of each centroid is determined by convolving the horizontal slice into a predefined window and then searching for the position of the maximum result of the convolution within a certain lateral range. This lateral range of the left centroids might for example be defined as the left half of the horizontal slice. However, to increase robustness, I narrowed the slice by disregarding the areas closest to the left and right borders of the image. The width that is disregarded is defined by the parameter *lateralBuffer* which is set on line 261 of helperfunctions.pt
Note: The window used for the convolution does not consist of uniform values (e.g.: all 1s)for the following reason: If the slice only contains a blob of points (that presumably belong to the lane markers) and this blob is smaller than the window, then the convolution yields the same results for several lateral positions of the window. Choosing a window with greater values at its  center yields a well defined peak in the result of the convolution, thus allowing us to zero in on the center of the blob.

Given that the lane markers are not always solid (lines can be dashed) it is possible that there isn't enough lane line content in each slice to robustly identify the lane marker. In those cases, the convolution method might "latch on" to noise (e.g. shadows that weren't excluded during the thresholding).
To avoid the negative impact of this behavior on our lane-line polynomials, I dismiss all centroids that are generated on the basis of a "low-yield" convolution. In other words, if the maximum value the convolution yields is below a certain threshold (*convThresh* defined on line 260) the resulting centroid will be disregarded. The actual removal of these centroids is done in the *excludeLowYieldCentroids* function which is defined on line 458 and deployed before the fitting of the polynomials

While the above describes how Method 1 calculates the positions of the centroids for **most** slices, the first (bottom) centroid (on each side) is an exception. Here, instead of defining the slice-height as the image-height divided by the number of slices, the slice height (separate for left and right) is a fraction of the image height defined by the variables *imfract_left* and *imfract_right*. These fractions are initially set by the user (see line 276) but if the maximu values of the resulting convolutions are below the threshold, these fractions are increased step-wise until the maximum value of the convolution exceeds the threshold value.

**Method 2**

Method 2 is called *find_window_centroids_nextFrame* and is defined in helperFunctions on line 340.
While it is very similar to Method 1, there are two main differences:
1. The heights of **all** slices(levels) are determined as the image-height divided by the number of slices (no exception for the bottom slice)
2. The search for the highest value of the convolution is restricted to a certain area around the lines defined by the lane line polynomials of the last frame (timestep). The width of that area is defined by the *margin* parameter

![alt text][image6]
Boxes drawn around identified centroids


**Excluding questionable centroids**
As mentioned above, centroids of questionable validity are removed using the *excludeLowYieldCentroids* function (line 458). This function is envoked in advancedLaneFind.py on line 102 - after the centroids have been identified, but before the polynomals are fitted. 

![alt text][image7]
Example of situation in which two centroids for the left line were removed due to low yield



**Fitting the polynomials**
Once the questionable centroids have been removed, the remaining centroids are used to fit a quadradic polynomial function. This is done to obtain continuous lane lines as well as to facilitate calculating the lanes' geometric properties.See the *fitLanePolynimial* function on line 153 of helperfunctions.py. 

Because we know that the vehicle is (almost) parallel to the lane lines, I am forcing the gradient of the polynomial function to be zero at the bottom of the image. This is achieved by mirroring the centroid data around the bottom edge of the image. (See line 173 and below)

Also, I wanted to make sure the polynomial has a minimal error at the bottom of the image (close to the car). I therefore created two duplicates of the bottom-most left and right centroids before fitting the polynomial (See line 161 and below).

![alt text][image8]
Example of polynomials fitted through centroids


**Checking the validity of the resulting lane lines**
I also implemented a function (*checkLaneValidity* on line 537) that keeps an eye on whether the results make sense. This function is used to help decide which method (see method 1 and method 2 above) to use and whether or not to use the latest results.

Currently the function only checks if the calculated width of the lane at the far end doesn't exceed a certain maximum value. (This would indicate that the line search algorithm latched on to objects outside the lane which might entice the vehicle to deviate into dangerous territory)

If the validity check returns *False*, then the most recent valid polynomials are reused. (advancedLaneFind.py line 114)


**When to use which method**
Method 1 is only used for the very first frame and for every frame that follows a negative validity check. All other frames are calculated with Method 2. 

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

These results are generated by the function *calcCurvAndGap* which is defined on line 469 of helperFunctions.py and called on line 139 of advancedLaneFind.py

The absolute value of the Radius of the curvature of each of the two lane lines is accomplished a by evaluating following formula at "y" equalling the real world (meters) value corresponding to the bottom end edge of the image (closest to the car).

R ​curve​​ =​((1+(2Ay+B)​^2^)​^3/2^)/(2A)
Where:
A is the coefficient of the quadratic term of the polynomial that represents the left or right lane
B is the coefficient of the linear term of the polynomial that represents the left or right lane

The *direction* of the curvature is determined by the sign of the coefficient quadratic term of the polynomial.

the lateral position of the vehicle with respect to the centerline of the lane is determined by calculating the lateral positions of the left and right lanes at the bottom of the image in real world terms (scaled from pixels to meters) and calculating the distance of their midpoints to the midpoint of the image (which is also the lateral center of the car) in real world terms.
​​ 
​​ 


####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
calculating the superimposed image is accomplished in the function *drawLanes_retransformed* which is called on line 142 of advancedLaneFind.py and defined on line 495 of helperFunctions.py.

![alt text][image9]
Example with lane area superimposed

![alt text][image10]
Example with geometric information

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_annotated_1.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The greates weakness of the current pipeline still lies in the limitations of the thresholding. Changing lighting conditions as well as shadows can still lead to false results. Also, what would happen if other cars drive in or cross into/out of the lane in front of the camera?

An interesting problem is also created by the roughness/bumpiness of the road.
The vehicles' pitch and heave motion (caused by bumps) leads to rapid changes in actual perspective. This means that the actual (real world) distance that falls into the region of interst as defined in the perspective transformation changes abruptly from image to image. As a result the distance (in pixels!) between the far ends of the two lane lines also changes abruptly. A narrow margin (area around the prevous results in which the algoritm searches for the lane lines) however prevents large changes in calculated lane line positions from frame to frame. as a result the calculated lane lines often deviate somewhat from the actual lane lines. However, increasing the margin will yield to situations in which the wrong features are mistaken for lane lines.

This issue could be addressed by caclulating the change in vehicle pitch from image to image and by correcting for that. Changes in pitch could be calculated by identifying the position of the horizon (specific thresholding would be required) and passing it through a high-pass-filter. 

