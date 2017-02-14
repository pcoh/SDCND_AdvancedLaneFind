import matplotlib
matplotlib.use('TKAgg')
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math


def calibrateCamera(calImages, nx, ny):
	print('number of calibration images:', len(calImages))
	objPoints = []
	imgPoints = []
	objP = np.zeros((nx*ny,3), np.float32)
	objP[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

	# find image points and append to array:
	for fname in calImages:
		calImg = mpimg.imread(fname)
		gray = cv2.cvtColor(calImg, cv2.COLOR_RGB2GRAY)
		ret, corners = cv2.findChessboardCorners(gray, (nx, ny),None)

		if ret==True:
			imgPoints.append(corners)
			objPoints.append(objP)

			img = cv2.drawChessboardCorners(calImg, (nx,ny), corners, ret)
			
	# Calculate camera matrix and distortion coeffs:
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1],None,None)
	return mtx, dist

def abs_sobel_thresh(channel, orient='x', sobel_kernel=3, thresh=(0, 255)):
	# Calculate directional gradient
	if orient =='x':
		sobel = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	elif orient =='y':
		sobel= cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

	absSobel = np.absolute(sobel)
	scaled = absSobel/np.max(absSobel)*255
	scaled_uint8 = np.uint8(scaled)
	grad_binary = np.zeros_like(scaled_uint8)
	grad_binary[(scaled_uint8>=thresh[0]) & (scaled_uint8 <=thresh[1])] = 1
	return grad_binary

def mag_sobel_thresh(channel, sobel_kernel=3, thresh=(0, 255)):
	# Calculate gradient magnitude
	# Apply threshold
	sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	mag = np.sqrt(np.square(sobelx) + np.square(sobely))
	scaled = np.uint8(mag/np.max(mag)*255)
	mag_binary = np.zeros_like(scaled)
	mag_binary[(scaled>thresh[0]) & (scaled<thresh[1])] = 1
	return mag_binary

def dir_threshold(channel, sobel_kernel=3, thresh=(0, np.pi/2)):
	# Calculate gradient direction
	# Apply threshold
	sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	
	sobelx_abs = np.absolute(sobelx)
	sobely_abs = np.absolute(sobely)
	gradDir = np.arctan2(sobely_abs, sobelx_abs)
	dir_binary = np.zeros_like(gradDir)
	dir_binary[(gradDir>=thresh[0]) & (gradDir<=thresh[1])] = 1
	return dir_binary

def color_threshold(channel, thresh=(170, 255)):
	channel = np.uint8(channel/np.max(channel)*255)
	color_binary = np.zeros_like(channel)
	color_binary[(channel>=thresh[0]) & (channel<=thresh[1])] = 1

	return color_binary

def thresholding(image, abs_thresh, mag_thresh, dir_thresh, R_thresh, S_thresh):
	R_channel = image[:,:,0]
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)

	H_channel = hls[:,:,0]
	S_channel = hls[:,:,2]

	grad_x_bin_gray = abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=abs_thresh)
	# grad_y_bin_gray = abs_sobel_thresh(gray, orient='y', sobel_kernel=3, thresh=(20, 100))
	grad_mag_bin_gray = mag_sobel_thresh(gray, sobel_kernel=9, thresh=mag_thresh)
	grad_dir_bin_gray = dir_threshold(gray, sobel_kernel=19, thresh = dir_thresh)

	grad_comb_bin_gray = np.zeros_like(grad_dir_bin_gray)
	grad_comb_bin_gray[(grad_x_bin_gray==1) | ((grad_mag_bin_gray == 1) & (grad_dir_bin_gray == 1))] = 1

	R_thresh_bin = color_threshold(R_channel, thresh=R_thresh)
	S_thresh_bin = color_threshold(S_channel, thresh=S_thresh)

	color_comb_bin = np.zeros_like(S_thresh_bin)
	color_comb_bin[(R_thresh_bin==1) | (S_thresh_bin==1)]=1

	total_comb_bin = np.zeros_like(grad_dir_bin_gray)
	total_comb_bin[(grad_comb_bin_gray==1) | (color_comb_bin ==1)] = 1

	return total_comb_bin

def transformPerspective(perspective_img):
	src = np.float32([[211,718.85],[603.97,445],[676.54,445],[1100,718.85]])
	# src = np.float32([[211,718.85],[571,468],[712.1,468],[1100,718.85]])
	
	dst = np.float32([[361,718.85],[361,0],[950,0],[950,718.85]])
	M = cv2.getPerspectiveTransform(src, dst)
	Minv = cv2.getPerspectiveTransform(dst, src)

	imSize = (perspective_img.shape[1],perspective_img.shape[0] )
	transformed_img = cv2.warpPerspective(perspective_img, M, imSize)
	
	return transformed_img, Minv


def fitLanePolynomial(leftx, lefty, rightx, righty, scaleX=1, scaleY=1):
	
	lefty_scaled = np.array(lefty)*scaleY	
	leftx_scaled = np.array(leftx)*scaleX

	righty_scaled = np.array(righty)*scaleY	
	rightx_scaled = np.array(rightx)*scaleX
	
	# duplicate the first item twice to put more weight on the areas closest to the car:
	lefty_scaled = np.append(lefty_scaled[0], lefty_scaled)
	leftx_scaled = np.append(leftx_scaled[0], leftx_scaled)
	righty_scaled = np.append(righty_scaled[0], righty_scaled)
	rightx_scaled = np.append(rightx_scaled[0], rightx_scaled)

	lefty_scaled = np.append(lefty_scaled[0], lefty_scaled)
	leftx_scaled = np.append(leftx_scaled[0], leftx_scaled)
	righty_scaled = np.append(righty_scaled[0], righty_scaled)
	rightx_scaled = np.append(rightx_scaled[0], rightx_scaled)


	# mirror the data around the bottom edge of the image to force the gradient to be 0 at the bottom edge
	lefty_scaled = np.append(lefty_scaled+np.amax(lefty_scaled), lefty_scaled)
	leftx_scaled = np.append(np.flipud(leftx_scaled), leftx_scaled)

	righty_scaled = np.append(righty_scaled+np.amax(righty_scaled), righty_scaled)
	rightx_scaled = np.append(np.flipud(rightx_scaled), rightx_scaled)

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty_scaled, leftx_scaled, 2)
	right_fit = np.polyfit(righty_scaled, rightx_scaled, 2)

	return left_fit, right_fit

def evaluateFitX(fitCoeffs, yaxis):
    fitx = fitCoeffs[0]*yaxis**2 + fitCoeffs[1]*yaxis + fitCoeffs[2]
    return fitx

def plotPoly_initialFrame(binary_warped,left_fit, right_fit, out_img, left_lane_inds, right_lane_inds, nonzerox, nonzeroy):
	ploty = createPlotYAxis(binary_warped)
	left_fitx =  evaluateFitX(left_fit, ploty)
	right_fitx =  evaluateFitX(rightt_fit, ploty)

	# left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	# right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	# ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	# plt.imshow(out_img)
	# plt.plot(left_fitx, ploty, color='yellow')
	# plt.plot(right_fitx, ploty, color='yellow')
	# plt.xlim(0, 1280)
	# plt.ylim(720, 0)
	# plt.show()

	return left_fitx, right_fitx, ploty, out_img

def createPlotYAxis(binImage):
	ploty = np.linspace(0, binImage.shape[0]-1, binImage.shape[0])
	return ploty

def plotPoly_nextFrame(binary_warped, left_fit, right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy, margin = 100):
	# ploty = createPlotYAxis(binary_warped)
	# left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	# right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	ploty = createPlotYAxis(binary_warped)
	left_fitx =  evaluateFitX(left_fit, ploty)
	right_fitx =  evaluateFitX(rightt_fit, ploty)

	# out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
	out_img = np.uint8(np.dstack((binary_warped, binary_warped, binary_warped))*255)
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
	output = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
	plt.imshow(output)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)
	plt.show()

	return left_fitx, right_fitx, output

def window_mask(width, height, img_ref, center,yCenter):
	output = np.zeros_like(img_ref)
	# output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
	output[int(yCenter-height/2):int(yCenter+height/2),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
	return output

def find_window_centroids(image, window_width, window_height, margin, mode='full'):
	convThresh = 60
	lateralBuffer = 100
	window_centroids_left = [] 
	window_centroids_right = []
	# Create window template that has higher values at center convolution properly centers around patches that are smaller than the window as well 
	if (window_width % 2 == 0):
		window_width = window_width + 1
	window = np.ones(window_width)*0.5
	window[math.ceil(window_width/2)] = 1
	
	# First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
	# and then np.convolve the vertical image slice with the window template     
	# Sum quarter bottom of image to get slice, could use a different ratio
	offset = window_width/2
	
	imfract = 0.2
	imfract_left = imfract
	imfract_right = imfract
	convVal_left = 0
	convVal_right = 0
	while convVal_left < convThresh:
		l_sum = np.sum(image[int(image.shape[0]*(1-imfract_left)):,lateralBuffer:int(image.shape[1]/2)], axis=0)
		convVal_left = np.amax(np.convolve(window,l_sum, mode))
		imfract_left = imfract_left+0.01
	l_center = np.argmax(np.convolve(window,l_sum, mode))-offset+lateralBuffer

	while convVal_right < convThresh:
		r_sum = np.sum(image[int(image.shape[0]*(1-imfract_right)):,int(image.shape[1]/2):image.shape[1]-lateralBuffer], axis=0)
		convVal_right = np.amax(np.convolve(window,r_sum, mode))
		imfract_right = imfract_right+0.01
	r_center = np.argmax(np.convolve(window,r_sum, mode))-offset+int(image.shape[1]/2)
	# print('convVal_right:',convVal_right)
	# print('imfract_right:',imfract_right)
	
	# print('l_center', l_center)
	# Add what we found for the first layer
	window_centroids_left.append(l_center)
	window_centroids_right.append(r_center)

	remainInd_left = [0]
	remainInd_right = [0]


	# lmax = [np.amax(np.convolve(window,l_sum, mode))]
	# rmax = [np.amax(np.convolve(window,r_sum, mode))]
	
	# Go through each layer looking for max pixel locations
	for level in range(1,(int)(image.shape[0]/window_height)):
		# convolve the window into the vertical slice of the image
		image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
		conv_signal = np.convolve(window, image_layer)
		# Find the best left centroid by using past left center as a reference
		# Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window

		l_min_index = int(max(l_center+offset-margin,0))
		l_max_index = int(min(l_center+offset+margin,image.shape[1]))
		
		#Consider the Centroid only if the convolution yields a value above the threshold:
		if np.amax(conv_signal[l_min_index:l_max_index]) > convThresh:
			l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
			remainInd_left.append(level)
		
		
		# Find the best right centroid by using past right center as a reference
		r_min_index = int(max(r_center+offset-margin,0))
		r_max_index = int(min(r_center+offset+margin,image.shape[1]))	

		#Consider the Centroid only if the convolution yields a value above the threshold:
		if np.amax(conv_signal[r_min_index:r_max_index]) > convThresh:
			r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
			remainInd_right.append(level)

		# Add what we found for that layer
		window_centroids_left.append(l_center)
		window_centroids_right.append(r_center)

		# lmax.append(np.amax(conv_signal[l_min_index:l_max_index]))
		# rmax.append(np.amax(conv_signal[r_min_index:r_max_index]))

		
	return window_centroids_left, window_centroids_right, remainInd_left, remainInd_right


def find_window_centroids_nextFrame(image, window_width, window_height, left_fit, right_fit, margin, mode='full'):
	convThresh = 60
	lateralBuffer = 100
	window_centroids_left = [] 
	window_centroids_right = []

	ploty = createPlotYAxis(image)
	
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


	# Create window template that has higher values at center convolution properly centers around patches that are smaller than the window as well 
	if (window_width % 2 == 0):
		window_width = window_width + 1
	window = np.ones(window_width)*0.5
	window[math.ceil(window_width/2)] = 1
	
	# First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
	# and then np.convolve the vertical image slice with the window template     
	# Sum quarter bottom of image to get slice, could use a different ratio
	offset = window_width/2
	
	
	remainInd_left = []
	remainInd_right = []

	
	# Go through each layer looking for max pixel locations
	for level in range(0,(int)(image.shape[0]/window_height)):
		# convolve the window into the vertical slice of the image

		image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
		conv_signal = np.convolve(window, image_layer)	


		# Find the best left centroid by using past left center as a reference
		# Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
		currY = int(image.shape[0]-(level+0.5)*window_height)
		startx_l = left_fit[0]*currY**2 + left_fit[1]*currY + left_fit[2]
		startx_r = right_fit[0]*currY**2 + right_fit[1]*currY + right_fit[2]

		l_min_index = int(max(startx_l+offset-margin,0))
		l_max_index = int(max(min(startx_l+offset+margin,image.shape[1]),margin))
		
		#Consider the Centroid only if the convolution yields a value above the threshold:
		if np.amax(conv_signal[l_min_index:l_max_index]) > convThresh:
			l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
			remainInd_left.append(level)
		else:
			# Dummy value since this centroid will be removed in "excludeLowYieldCentroids"
			l_center = image.shape[1]		
		
		# Find the best right centroid by using past right center as a reference
		r_min_index = int(min(max(startx_r+offset-margin,0),image.shape[1]-margin))
		r_max_index = int(min(startx_r+offset+margin,image.shape[1]))	
		# print('r_min_index: ',r_min_index)
		# print('r_max_index: ',r_max_index)

		#Consider the Centroid only if the convolution yields a value above the threshold:
		if np.amax(conv_signal[r_min_index:r_max_index]) > convThresh:
			r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
			remainInd_right.append(level)
		else:
			# Dummy value since this centroid will be removed in "excludeLowYieldCentroids"
			r_center = 0

		# Add what we found for that layer
		window_centroids_left.append(l_center)
		window_centroids_right.append(r_center)
		
	return window_centroids_left, window_centroids_right, remainInd_left, remainInd_right

def plotCentroidBoxes(warped,window_centroids_left, window_centroids_right,window_width, window_height, yCenters_left, yCenters_right, left_fitx, right_fitx):
	# If we found any window centers
	# Points used to draw all the left and right windows
	l_points = np.zeros_like(warped)
	r_points = np.zeros_like(warped)

	if len(window_centroids_left) > 0:	   
		# Go through each level and draw the windows 	
		for level in range(0,len(window_centroids_left)):
			# Window_mask is a function to draw window areas
			# l_mask = window_mask(window_width,window_height,warped,window_centroids_left[level],level)
			l_mask = window_mask(window_width,window_height,warped,window_centroids_left[level],yCenters_left[level])
			# Add graphic points from window mask here to total pixels found 
			l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255

	if len(window_centroids_right) > 0:
		# Go through each level and draw the windows
		for level in range(0,len(window_centroids_right)):
			# Window_mask is a function to draw window areas
			# r_mask = window_mask(window_width,window_height,warped,window_centroids_right[level],level)
			r_mask = window_mask(window_width,window_height,warped,window_centroids_right[level],yCenters_right[level])
			# Add graphic points from window mask here to total pixels found 
			r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

	if ((len(window_centroids_left) > 0) | (len(window_centroids_right) > 0)):
		# Draw the results
		template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
		zero_channel = np.zeros_like(template) # create a zero color channle 
		template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
		warpage = np.array(cv2.merge((warped*255,warped*255,warped*255)),np.uint8) # making the original road pixels 3 color channels
		centroidImg = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results	    

	# If no window centers found, just display orginal road image
	else:
		centroidImg = np.array(cv2.merge((warped,warped,warped)),np.uint8)

	
	# ploty = createPlotYAxis(warped)
	# # Display the final results
	# plt.imshow(centroidImg)
	# plt.title('window fitting results')
	# plt.plot(left_fitx, ploty, color='yellow')
	# plt.plot(right_fitx, ploty, color='yellow')
	# plt.xlim(0, 1280)
	# plt.ylim(720, 0)
	# plt.show()

	return centroidImg


def excludeLowYieldCentroids(window_centroids_left, window_centroids_right, yCenters, remainInd_left, remainInd_right):
	window_centroids_left = np.array(window_centroids_left)	
	window_centroids_left = window_centroids_left[remainInd_left]
	yCenters_left = yCenters[remainInd_left]

	window_centroids_right = np.array(window_centroids_right)
	window_centroids_right = window_centroids_right[remainInd_right]
	yCenters_right = yCenters[remainInd_right]


	return window_centroids_left, window_centroids_right, yCenters_left, yCenters_right 

def calcCurvAndGap(left_fit_scaled, right_fit_scaled, y_eval_scaled, imWidth_scaled):	
	left_curverad = ((1 + (2*left_fit_scaled[0]*y_eval_scaled + left_fit_scaled[1])**2)**1.5) / np.absolute(2*left_fit_scaled[0])
	right_curverad = ((1 + (2*right_fit_scaled[0]*y_eval_scaled + right_fit_scaled[1])**2)**1.5) / np.absolute(2*right_fit_scaled[0])

	# determine direction of turn by looking at the sign of the second derivative of the fit:
	left_curverad = left_curverad*np.sign(left_fit_scaled[0])
	right_curverad = right_curverad*np.sign(right_fit_scaled[0])

	left_curverad = round(left_curverad,1)
	right_curverad = round(right_curverad,1)

	
	left_latPos  = left_fit_scaled[0]*y_eval_scaled**2 + left_fit_scaled[1]*y_eval_scaled + left_fit_scaled[2]
	right_latPos = right_fit_scaled[0]*y_eval_scaled**2 + right_fit_scaled[1]*y_eval_scaled + right_fit_scaled[2]
	
	laneCenter = (left_latPos+right_latPos)/2
	vehicleCenter = imWidth_scaled/2
	

	driverGap = round(vehicleCenter -laneCenter,2) # positive if vehicle too far right

	return left_curverad, right_curverad, driverGap


def drawLanes_retransformed(undist, fitImg, Minv, left_fitx, right_fitx, left_curverad, right_curverad, driverGap):
	# Create an image to draw the lines on
	warp_zero = np.zeros_like(fitImg[:,:,0]).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
	ploty = createPlotYAxis(warp_zero)

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
	# Combine the result with the original image
	resultImg = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
	# plt.imshow(resultImg)
	# plt.show()


	# f, axes = plt.subplots(1, 2, figsize=(15, 12))
	# font_size = 10
	
	# axes[0].imshow(resultImg)
	# axes[0].set_title('augmented', fontsize=font_size)
	
	# axes[0].text(0.05, 0.9,'Left Radius: %sm' % (left_curverad), ha='left', va='center', transform=axes[0].transAxes)
	# axes[0].text(0.05, 0.84,'Right Radius: %sm' % (right_curverad), ha='left', va='center', transform=axes[0].transAxes)
	# axes[0].text(0.05, 0.78,'Deviation: %sm' % (driverGap), ha='left', va='center', transform=axes[0].transAxes)

	# axes[1].imshow(fitImg, cmap='gray')
	# axes[1].set_title('topView fit', fontsize=font_size)
	# plt.plot(left_fitx, ploty, color='yellow')
	# plt.plot(right_fitx, ploty, color='yellow')
	# plt.xlim(0, 1280)
	# plt.ylim(720, 0)
	# plt.show()

	return resultImg

def checkLaneValidity(left_fit_scaled, right_fit_scaled, y_eval_scaled, imWidth_scaled, trueLaneWidth):
	currLaneWidth_far = round(evaluateFitX(right_fit_scaled, 0) - evaluateFitX(left_fit_scaled, 0), 2)
	currLaneWidth_near = round(evaluateFitX(right_fit_scaled, y_eval_scaled) - evaluateFitX(left_fit_scaled, y_eval_scaled),2)
	# print("lanewidth Far: ", currLaneWidth_far)
	# print("lanewidth Close: ", currLaneWidth_near)
	# currLaneWidth_far = evaluateFitX(left_fit_scaled, y_eval_scaled)
	if currLaneWidth_far > 6.4:
		laneValid = False
	else:
		laneValid = True
	return laneValid, currLaneWidth_far, currLaneWidth_near

# def reduceNoise(bin_Image, kernelSize=20, filledRatio = 0.15):
# 	nonzero = bin_Image.nonzero()
# 	nonzeroy = nonzero[0]
# 	nonzerox = nonzero[1]
# 	imgHeight = bin_Image.shape[0]
# 	imgWidth = bin_Image.shape[1]
# 	print("Nonzeros length", len(nonzeroy))
# 	halfkernel = kernelSize/2
# 	clean_Image = np.copy(bin_Image)
# 	for idx in range(len(nonzeroy)):
# 		window = bin_Image[np.amax((0,nonzeroy[idx]-halfkernel)):np.amin((imgHeight,nonzeroy[idx]+halfkernel)), np.amax((0,nonzerox[idx]-halfkernel)):np.amin((imgWidth, nonzerox[idx]+halfkernel))]
# 		windowSize = window.size
# 		windowSum = np.sum(window)
# 		if windowSum < windowSize*filledRatio:
# 			clean_Image[nonzeroy[idx], nonzerox[idx]]= 0
# 	return clean_Image

# def detectLaneLines_initialFrame(topView_img, nwindows = 9, margin = 80):
# 	# Take a histogram of the bottom half of the image

# 	histogram = np.sum(topView_img[np.uint8(math.ceil(topView_img.shape[0]/2)):,:], axis=0)
# 	# Create an output image to draw on and  visualize the result
# 	out_img = np.uint8(np.dstack((topView_img, topView_img, topView_img))*255)
	
# 	# Find the peak of the left and right halves of the histogram
# 	# These will be the starting point for the left and right lines
# 	midpoint = np.int(histogram.shape[0]/2)
# 	leftx_base = np.argmax(histogram[:midpoint])
# 	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

# 	# Set height of windows
# 	window_height = np.int(topView_img.shape[0]/nwindows)
# 	# Identify the x and y positions of all nonzero pixels in the image
# 	nonzero = topView_img.nonzero()
# 	nonzeroy = np.array(nonzero[0])
# 	nonzerox = np.array(nonzero[1])
# 	# Current positions to be updated for each window
# 	leftx_current = leftx_base
# 	rightx_current = rightx_base
	
# 	# Set minimum number of pixels found to recenter window
# 	minpix = 50
# 	# Create empty lists to receive left and right lane pixel indices
# 	left_lane_inds = []
# 	right_lane_inds = []

# 	# Step through the windows one by one
# 	for window in range(nwindows):
# 		# Identify window boundaries in x and y (and right and left)
# 		win_y_low = topView_img.shape[0] - (window+1)*window_height
# 		win_y_high = topView_img.shape[0] - window*window_height
# 		win_xleft_low = leftx_current - margin
# 		win_xleft_high = leftx_current + margin
# 		win_xright_low = rightx_current - margin
# 		win_xright_high = rightx_current + margin
# 		# Draw the windows on the visualization image
# 		cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
# 		cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
# 		# Identify the nonzero pixels in x and y within the window
# 		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
# 		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
# 		# Append these indices to the lists
# 		left_lane_inds.append(good_left_inds)
# 		right_lane_inds.append(good_right_inds)
# 		# If you found > minpix pixels, recenter next window on their mean position
# 		if len(good_left_inds) > minpix:
# 			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
# 		if len(good_right_inds) > minpix:        
# 			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

# 	# Concatenate the arrays of indices
# 	left_lane_inds = np.concatenate(left_lane_inds)
# 	right_lane_inds = np.concatenate(right_lane_inds)

# 	# Extract left and right line pixel positions
# 	leftx = nonzerox[left_lane_inds]
# 	lefty = nonzeroy[left_lane_inds] 
# 	rightx = nonzerox[right_lane_inds]
# 	righty = nonzeroy[right_lane_inds] 

# 	return leftx, lefty, rightx, righty,out_img, left_lane_inds, right_lane_inds, nonzerox, nonzeroy


# def detectLaneLines_nextFrame(binary_warped, left_fit, right_fit, margin = 50):
# 	nonzero = binary_warped.nonzero()
# 	nonzeroy = np.array(nonzero[0])
# 	nonzerox = np.array(nonzero[1])
# 	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
# 	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

# 	# Again, extract left and right line pixel positions
# 	leftx = nonzerox[left_lane_inds]
# 	lefty = nonzeroy[left_lane_inds] 
# 	rightx = nonzerox[right_lane_inds]
# 	righty = nonzeroy[right_lane_inds]
# 	# Fit a second order polynomial to each
# 	# left_fit = np.polyfit(lefty, leftx, 2)
# 	# right_fit = np.polyfit(righty, rightx, 2)
# 	# Generate x and y values for plotting
# 	# ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
# 	# left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
# 	# right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

# 	return leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, nonzerox, nonzeroy
