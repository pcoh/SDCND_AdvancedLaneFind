import matplotlib
matplotlib.use('TKAgg')
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from helperFunctions import *
import time
import pylab
import imageio

calImages = glob.glob('camera_cal/calibration*.jpg')
nx = 9
ny = 6
trueLaneWidth = 3.7 # expected width in meters

# Tuning parameters:
openkernelSize = 7
window_width = 100
window_height = 80
ym_per_pix = 35/720 # meters per pixel in y dimension
xm_per_pix = trueLaneWidth/(950-361) # meters per pixel in x dimension

videoVersion = 1
sourceFileName = 'project_video.mp4'
# sourceFileName = 'challenge_video.mp4'
vid = imageio.get_reader(sourceFileName,  'ffmpeg')
num_frames=vid._meta['nframes']
framerate = vid.get_meta_data()['fps']
print("source video frame rate: ", framerate)
print('number of frames in video: ',num_frames)

#Choose which frames of the video to consider:
# frames = [100,101,102,103,104]
frames = [933]
# frames = np.arange(0,num_frames,1)
# frames = np.arange(0,192,1)

# calibrate the camera using the provided calibration images:
mtx, dist = calibrateCamera(calImages, nx, ny)

# initialize a video file to save the results to:
targetFileName = sourceFileName[:-4]+'_annotated_'+str(videoVersion)+'.mp4'
writer = imageio.get_writer(targetFileName, fps=framerate)

# Initially, look for lane lines in the entire image:
fullSearch = True

# initialize the fit parameters to use when found lane lines are invalid (with dummy values):
left_fit_prev = [0,1,2]
right_fit_prev = [0,1,2]
left_fit_scaled_prev = [0,1,2]
right_fit_scaled_prev = [0,1,2]


# Loop through frames of video:
for frame in frames:
    print('Processing frame ', frame)
    currImg = vid.get_data(frame)

    # plt.imshow(currImg)
    # plt.show()

    # Correction (undistort) image:
    undistImg = cv2.undistort(currImg, mtx, dist, None, mtx)

    # plt.imshow(undistImg)
    # plt.show()

    # Apply Color/gradient thresholding:
    thresh_img = thresholding(undistImg, abs_thresh = (20, 100), mag_thresh = (30, 100), dir_thresh = (0.7, 1.3), R_thresh = (220, 255), S_thresh = (170,240))

    # plt.imshow(thresh_img, cmap="gray")
    # plt.show()

    # Perform perspective transform to obtain top view:
    topView_img, Minv = transformPerspective(thresh_img)

    # plt.imshow(topView_img, cmap="gray")
    # plt.show()

    # Remove small specs (Noise) from top view image:
    kernel = np.ones((openkernelSize,openkernelSize), np.uint8)
    noiseFree_img = cv2.morphologyEx(topView_img, cv2.MORPH_OPEN, kernel)

    # calculate image width in meters
    imWidth_scaled = noiseFree_img.shape[1]*xm_per_pix

    #calculate y coordinates of window centroids
    yCenters = np.arange(noiseFree_img.shape[0],0,-window_height)-window_height/2
    
    if fullSearch:
        # Find lane centroids by searching the entire image area (Full Search):
        window_centroids_left, window_centroids_right, remainInd_left, remainInd_right = find_window_centroids(noiseFree_img, window_width, window_height, margin = 85)

        # In the next frame, perform search only in a target area: 
        fullSearch = False
    else:
        # Find lane centroids in search area derived from previous timestep
        window_centroids_left, window_centroids_right, remainInd_left, remainInd_right = find_window_centroids_nextFrame(noiseFree_img, window_width, window_height,left_fit, right_fit, margin = 20, mode='full')

    # Exclde centroids that might have latched on to noise rather than to actual portions of the lane markers
    window_centroids_left, window_centroids_right, yCenters_left, yCenters_right = excludeLowYieldCentroids(window_centroids_left, window_centroids_right, yCenters, remainInd_left, remainInd_right)
    
    # Fit polynomials to centroids to create continuouse lane borders(used for overlaying lane on camera image):
    left_fit, right_fit = fitLanePolynomial(window_centroids_left, yCenters_left, window_centroids_right, yCenters_right)
     # Fit polynomials to centroids to create continuouse lane borders (to enable calulation of curvatures and distance from center):
    left_fit_scaled, right_fit_scaled = fitLanePolynomial(window_centroids_left, yCenters_left, window_centroids_right, yCenters_right, scaleX=xm_per_pix, scaleY=ym_per_pix)
    
    # check if found lanes make sense:
    y_eval_scaled = noiseFree_img.shape[0]*ym_per_pix
    laneValid,currLaneWidth_far, currLaneWidth_near = checkLaneValidity(left_fit_scaled, right_fit_scaled, y_eval_scaled, imWidth_scaled, trueLaneWidth)
    
    # If the found lanes aren't valid, reuse the lanes from the previous frame and do a full search for the next timestep:
    if laneValid == False:
        left_fit = left_fit_prev
        right_fit = right_fit_prev
        left_fit_scaled = left_fit_scaled_prev
        right_fit_scaled = right_fit_scaled_prev
        fullSearch = True


    # Keep current fit results available for the next timestep (next frame):
    left_fit_prev = left_fit
    right_fit_prev = right_fit
    left_fit_scaled_prev =  left_fit_scaled
    right_fit_scaled_prev = right_fit_scaled 
    
    # Create plottable lane data by evaluating fits along the entire y-axis:
    ploty = createPlotYAxis(noiseFree_img)
    left_fitx =  evaluateFitX(left_fit, ploty)
    right_fitx = evaluateFitX(right_fit, ploty)


    # Plot boxes around the found centroids:
    fitImg = plotCentroidBoxes(noiseFree_img,window_centroids_left, window_centroids_right,window_width, window_height, yCenters_left, yCenters_right,left_fitx, right_fitx)


    #calculate lateral curve radius (in real world coordinates):
    left_curverad, right_curverad, driverGap = calcCurvAndGap(left_fit_scaled, right_fit_scaled, y_eval_scaled, imWidth_scaled)

    #Superimpose the found lane over the video frame:
    resultImg = drawLanes_retransformed(undistImg, fitImg, Minv, left_fitx, right_fitx, left_curverad, right_curverad, driverGap)
    
    # annotate the frame with geometric information (curve-radius, deviation of vehicle from centerline etc.)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(resultImg,'Left Radius: %sm' % (left_curverad),(10,50), font, 0.9,(255,138,0),2,cv2.LINE_AA)
    cv2.putText(resultImg,'Right Radius: %sm' % (right_curverad),(10,90), font, 0.9,(255,138,0),2,cv2.LINE_AA)
    cv2.putText(resultImg,'Lateral Deviation: %sm' % (driverGap),(10,130), font, 0.9,(255,138,0),2,cv2.LINE_AA)
    cv2.putText(resultImg,'Lane width @near: %sm' % (currLaneWidth_near),(10,170), font, 0.9,(255,138,0),2,cv2.LINE_AA)
    cv2.putText(resultImg,'Lane width @far: %sm' % (currLaneWidth_far),(10,210), font, 0.9,(255,138,0),2,cv2.LINE_AA)
    cv2.putText(resultImg,'Frame: %s' % (frame),(10,250), font, 0.9,(255,138,0),2,cv2.LINE_AA)

    plt.imshow(resultImg)
    plt.show()


   
    #append annotated frame to video:
    writer.append_data(resultImg)

#Finalize the video: 
writer.close()
