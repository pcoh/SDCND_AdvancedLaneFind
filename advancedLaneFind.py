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

videoVersion = 1
# sourceFileName = 'project_video.mp4'
sourceFileName = 'challenge_video.mp4'
vid = imageio.get_reader(sourceFileName,  'ffmpeg')
num_frames=vid._meta['nframes']
framerate = vid.get_meta_data()['fps']
# print("source video frame rate: ", framerate)
# print('number of frames in video: ',num_frames)
# frames = [996,997,998,999,1000,1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016]
# frames = [1007,1008,1009,1010]
# frames = [100,101,102,103,104]
# frames = [362]
frames = np.arange(0,num_frames,1)

frames = np.arange(0,192,1)

calImg = mpimg.imread('camera_cal/calibration1.jpg')
plt.imshow(calImg)
plt.show()

mtx, dist = calibrateCamera(calImages, nx, ny)

undistImg = cv2.undistort(calImg, mtx, dist, None, mtx)
plt.imshow(undistImg)
plt.show()

targetFileName = sourceFileName[:-4]+'_annotated_'+str(videoVersion)+'.mp4'
writer = imageio.get_writer(targetFileName, fps=framerate)

ym_per_pix = 35/720 # meters per pixel in y dimension
xm_per_pix = trueLaneWidth/(950-361) # meters per pixel in x dimension

fullSearch = True
left_fit_prev = [0,1,2]
right_fit_prev = [0,1,2]
left_fit_scaled_prev = [0,1,2]
right_fit_scaled_prev = [0,1,2]

for frame in frames:
    print('Processing frame ', frame)
    currImg = vid.get_data(frame)

    # Distortion correction:
    undistImg = cv2.undistort(currImg, mtx, dist, None, mtx)
    # plt.imshow(undistImg)
    # plt.show()
    # Apply Color/gradient thresholding:
    thresh_img = thresholding(undistImg, abs_thresh = (20, 100), mag_thresh = (30, 100), dir_thresh = (0.7, 1.3), R_thresh = (220, 255), S_thresh = (170,240))
    # Perspective transform
    topView_img, Minv = transformPerspective(thresh_img)

    
    kernel = np.ones((openkernelSize,openkernelSize), np.uint8)
    noiseFree_img = cv2.morphologyEx(topView_img, cv2.MORPH_OPEN, kernel)
    imWidth_scaled = noiseFree_img.shape[1]*xm_per_pix

    yCenters = np.arange(noiseFree_img.shape[0],0,-window_height)-window_height/2
    
    # Find lane centroids starting from scratch or in search area derived from previous timestep:
    if fullSearch:
        window_centroids_left, window_centroids_right, remainInd_left, remainInd_right = find_window_centroids(noiseFree_img, window_width, window_height, margin = 85)

        fullSearch = False
    else:
        # Find lane centroids starting from scratch or in search area derived from previous timestep
        window_centroids_left, window_centroids_right, remainInd_left, remainInd_right = find_window_centroids_nextFrame(noiseFree_img, window_width, window_height,left_fit, right_fit, margin = 20, mode='full')

    # Exclde centroids that might have latched on to small noise
    window_centroids_left, window_centroids_right, yCenters_left, yCenters_right = excludeLowYieldCentroids(window_centroids_left, window_centroids_right, yCenters, remainInd_left, remainInd_right)
    
    # Fit polynomials to centroids to create continuouse lanes (first for image augmentation then to enable calulation of curvatures and distance from center:
    left_fit, right_fit = fitLanePolynomial(window_centroids_left, yCenters_left, window_centroids_right, yCenters_right)
    left_fit_scaled, right_fit_scaled = fitLanePolynomial(window_centroids_left, yCenters_left, window_centroids_right, yCenters_right, scaleX=xm_per_pix, scaleY=ym_per_pix)
    
    # check if found lanes make sense:
    y_eval_scaled = noiseFree_img.shape[0]*ym_per_pix
    laneValid,currLaneWidth_far, currLaneWidth_near = checkLaneValidity(left_fit_scaled, right_fit_scaled, y_eval_scaled, imWidth_scaled, trueLaneWidth)
    
    if laneValid == False:
        left_fit = left_fit_prev
        right_fit = right_fit_prev
        left_fit_scaled = left_fit_scaled_prev
        right_fit_scaled = right_fit_scaled_prev
        fullSearch = True

    left_fit_prev = left_fit
    right_fit_prev = right_fit
    left_fit_scaled_prev =  left_fit_scaled
    right_fit_scaled_prev = right_fit_scaled 
    

    ploty = createPlotYAxis(noiseFree_img)
    left_fitx =  evaluateFitX(left_fit, ploty)
    right_fitx = evaluateFitX(right_fit, ploty)

    # left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    # right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    fitImg = plotCentroidBoxes(noiseFree_img,window_centroids_left, window_centroids_right,window_width, window_height, yCenters_left, yCenters_right,left_fitx, right_fitx)



    #calculate lateral curve radius:
    left_curverad, right_curverad, driverGap = calcCurvAndGap(left_fit_scaled, right_fit_scaled, y_eval_scaled, imWidth_scaled)

    resultImg = drawLanes_retransformed(undistImg, fitImg, Minv, left_fitx, right_fitx, left_curverad, right_curverad, driverGap)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(resultImg,'Left Radius: %sm' % (left_curverad),(10,50), font, 0.9,(255,138,0),2,cv2.LINE_AA)
    cv2.putText(resultImg,'Right Radius: %sm' % (right_curverad),(10,90), font, 0.9,(255,138,0),2,cv2.LINE_AA)
    cv2.putText(resultImg,'Lateral Deviation: %sm' % (driverGap),(10,130), font, 0.9,(255,138,0),2,cv2.LINE_AA)
    cv2.putText(resultImg,'Lane width @near: %sm' % (currLaneWidth_near),(10,170), font, 0.9,(255,138,0),2,cv2.LINE_AA)
    cv2.putText(resultImg,'Lane width @far: %sm' % (currLaneWidth_far),(10,210), font, 0.9,(255,138,0),2,cv2.LINE_AA)
    cv2.putText(resultImg,'Frame: %s' % (frame),(10,250), font, 0.9,(255,138,0),2,cv2.LINE_AA)
   
    writer.append_data(resultImg)

    
writer.close()
