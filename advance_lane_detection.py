import numpy as np
import cv2
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from moviepy.editor import ImageSequenceClip
from functools import partial

def display_output(img1,img1_title,img2,img2_title):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    if img2_title == 'Peaks':
        ax1.imshow(img1, cmap = 'gray')
        ax1.set_title(img1_title, fontsize=30)
        plt.plot(img2)
        ax2.set_title(img2_title, fontsize=30)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    else:
        ax1.imshow(img1)
        ax1.set_title(img1_title, fontsize=30)
        ax2.imshow(img2, cmap='gray')
        ax2.set_title(img2_title, fontsize=30)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def camera_calib(img):
    nx = 9 #inside corners along x axis
    ny = 6 #inside corners along y axis
    canvas = np.copy(img)
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    #Convert to grayscale
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    #Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
    #print(ret)
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        cv2.drawChessboardCorners(canvas, (nx,ny), corners, ret)
        #plt.imshow(img)
        #plt.show()

    # Camera calibration, given object points, image points, and the shape of the grayscale image,
    # returns distortion coefficients (dist) camera matrix (mtx), camera position in the world, rotation (rvecs), translation (tvecs)

    ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints, gray.shape[::-1], None,None)
    return mtx,dist,nx,ny

def distort_correction(img, mtx,dist):
    undist_image = cv2.undistort(img,mtx,dist,None,None)
    display_output(img,'Original image',undist_image,'undistorted_image')
    return undist_image

def warp(img,mtx,dist,src_points,dst_points):
    img_size = (img.shape[1], img.shape[0]) # width x height
    # Undistort using mtx and dist
    undist = distort_correction(img,mtx,dist)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src_points,dst_points)
    # use.cv2.getPerspectiveTransform() to get Minv, the inverse transform matrix
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(undist, M, img_size)

    display_output(img,'Original image',warped,'Warped image')

    return M,Minv,warped

def sobel_threshold(warped_img, orient, thresh_min=0, thresh_max=255, sobel_kernel = 3):
    # pass a single colour channel to the cv2.sobel() function, convert to grayscale
    gray = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY)
    #Check orientation and apply derivative
    if orient == 'x':
        der = np.absolute(cv2.Sobel(gray,cv2.CV_64F, 1, 0))
    if orient == 'y':
        der = np.absolute(cv2.Sobel(gray,cv2.CV_64F, 0, 1))

    #convert image to 8-bit
    scaled_der = np.uint8(255*der/np.max(der))

    binary_img = np.zeros_like(scaled_der)
    #Thresholding
    binary_img[(scaled_der>=thresh_min)&(scaled_der<=thresh_max)] =1
    
    display_output(warped_img,'Warped image',binary_img,f'Sobel_{orient}')
    return binary_img

def mag_der_thresh(warped, mag_thresh,sobel_kernel = 9):
    #convert to grayscale
    gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    # calulate the derivative in the x and y direction using Sobel operator
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    #rescale to 8-bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    display_output(warped,'Original image',binary_output,'gradient magnitude')
    return binary_output

def dir_der_thresh(warped,dir_thresh,sobel_kernel = 15):
    # Grayscale
    gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1])] = 1

    display_output(warped,'Original image',binary_output,'gradient direction')
    return binary_output

def hls_thresholding(warped, color_thresh_s = (0,255), color_thresh_l = (0,255)):
    #convert to hls colorspace
    hls = cv2.cvtColor(warped, cv2.COLOR_RGB2HLS) #RGB as we are using CV2
    #Isolate for saturation channel as its best at identifying lane lines in most scenarios (shadows)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    binary_output_s = np.zeros_like(s_channel)
    binary_output_l = np.zeros_like(l_channel)

    # apply thresholds to isolate lane lines
    binary_output_s[(s_channel > color_thresh_s[0]) & (s_channel <= color_thresh_s[1])] = 1
    binary_output_l[(l_channel > color_thresh_l[0]) & (l_channel <= color_thresh_l[1])] = 1
    display_output(warped,'Original image',binary_output_s,'S-channel Thresholding')
    display_output(warped,'Original image',binary_output_l,'L-channel Thresholding')
    return binary_output_s,binary_output_l

def hsv_thresholding(warped, color_thresh):
    #convert to hls colorspace
    hsv = cv2.cvtColor(warped, cv2.COLOR_RGB2HSV) #RGB as we are using CV2
    #Isolate for saturation channel as its best at identifying lane lines in most scenarios (shadows)
    h_channel = hsv[:,:,0]
    s_channel = hsv[:,:,1]
    v_channel = hsv[:,:,2]
    binary_output_v = np.zeros_like(v_channel)
    # apply thresholds to isolate lane lines
    binary_output_v[(v_channel > color_thresh[0]) & (v_channel <= color_thresh[1])] = 1

    display_output(warped,'Original image',binary_output_v,'HSV Thresholding')
    return binary_output_v

def combined_thresh(warped):
    gradient_x = sobel_threshold(warped,'x',thresh_min=20,thresh_max=200)
    gradient_y = sobel_threshold(warped,'y',thresh_min=20,thresh_max=200)
    magnitude_thresh = mag_der_thresh(warped,(30,100), sobel_kernel=7)
    direction_thresh = dir_der_thresh(warped,(1.3,1.6),sobel_kernel=15)
    combined = np.zeros_like(direction_thresh)
    combined[(gradient_x >= 0.5)|((magnitude_thresh >= 0.8)&(direction_thresh <= 0.5))]= 1
    # Plot the result
    #display_output(warped,'Original image',combined,'Combined Thresholding')

    s_binary,l_binary = hls_thresholding(warped,(115,255),(195,255))
    v_binary = hsv_thresholding(warped,(200,255))
    color_combined = np.zeros_like(direction_thresh)
    color_combined[(combined>=0.5)|(((s_binary>=0.5) & (v_binary>=0.5))|(l_binary>=0.5))] = 1
    display_output(warped,'Original image',color_combined,'Color Combined Thresholding')
    return color_combined

def histogram_filter(img):
    # Lane lines are likely to be mostly vertical nearest to the car
    # Hence, grab only the bottom half of the image
    bottom_half = img[img.shape[0]//2:,:]
    # Sum across image pixels vertically - make sure to set `axis`
    histogram = np.sum(bottom_half, axis=0)
    display_output(img,'Color combined Thresholding', histogram,'Peaks')
    #Histogram shape is (1280,)
    output_img = np.dstack((img,img,img))
    output_img = (output_img*255).astype('uint8')
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = histogram.shape[0]//2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9 # Choose the number of sliding windows
    margin = 100 # Set the width of the windows +/- margin
    minpix = 50 # Set minimum number of pixels found to recenter window    
    window_height = np.int8(img.shape[0]//nwindows) # Set height of windows - based on nwindows

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base #Intitial position (base of left lane)
    rightx_current = rightx_base #Initial position (base of right lane)

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height         #higher y coordinate of margin from top of the image
        win_y_high = img.shape[0] - window*window_height            #lower y coordinate of margin from top of the image

        #Find the four below boundaries of the window (width of the window is 200)
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(output_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 5) 
        cv2.rectangle(output_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 5) 

        # Identify the nonzero pixels in x and y within the window 
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        #recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = round(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = round(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    left_lane_inds = np.concatenate(left_lane_inds)
    #print(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    #print(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    #print(leftx)
    lefty = nonzeroy[left_lane_inds] 
    #print(lefty)
    rightx = nonzerox[right_lane_inds]
    #print(rightx)
    righty = nonzeroy[right_lane_inds]
    #print(righty)

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty 
    ## Visualization ##
    # Colors in the left and right lane regions
    output_img[lefty, leftx] = [255, 0, 0]
    output_img[righty, rightx] = [100, 200, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.imshow(output_img)
    plt.show()

    return output_img,left_fitx,right_fitx,ploty,left_fit,right_fit 

def fit_poly(img_shape,leftx, lefty, rightx, righty):
    
    # Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    
    # Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return ploty, left_fit, right_fit, left_fitx, right_fitx

def search_from_prior(color_combined, left_fit,right_fit):
    # Margin around polynomial to search
    margin = 100
    # Grab activated pixels
    nonzero = color_combined.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set the area of search based on activated x-values within the +/- margin of our polynomial function 
    left_lane_inds = ((nonzerox > 
                       (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)
                      ) & (nonzerox < 
                           (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    ploty,left_fit, right_fit, left_fitx, right_fitx = fit_poly(color_combined.shape, leftx, lefty, rightx, righty)
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = (np.dstack((color_combined, color_combined, color_combined))*255).astype('uint8')
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [100, 200, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color = 'yellow')
    plt.plot(right_fitx, ploty, color = 'yellow')
    plt.imshow(out_img)
    plt.show()
    ## End visualization steps ##
    return out_img,ploty,left_fitx,right_fitx,left_fit,right_fit

# generate coefficient values for lane datapoints in meters
def generate_data(ploty, left_fitx, right_fitx, ym_per_pix, xm_per_pix):
    
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    return ploty*ym_per_pix, left_fit_cr, right_fit_cr

def curvature(ploty, left_fitx, right_fitx):
    # define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    lane_centre = (left_fitx[-1] + right_fitx[-1])/2
    centre_offset_pixels = img_size[0]/2 - lane_centre
    # convert to metres from pixels using conversion
    centre_offset_metres = xm_per_pix*centre_offset_pixels
    
    # generate data points for left and right curverad
    ploty, left_fit_cr, right_fit_cr = generate_data(ploty, left_fitx, right_fitx, ym_per_pix, xm_per_pix)
    
    # define y-value where we want radius of curvature from the bottom of the image
    y_eval = np.max(ploty)
    
    ##### Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = ((1+(2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5)//(2*abs(left_fit_cr[0]))  ## Implement the calculation of the left line here
    right_curverad = ((1+(2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5)//(2*abs(right_fit_cr[0]))  ## Implement the calculation of the right line here
    average_curvature = (left_curverad + right_curverad)/2
    print("Average Curvature: " + str(average_curvature) + " m")
    print("Vehicle Offset from Centre of Lane: " + str(centre_offset_metres) + " m")

def draw_shade(img, warped, left_fit, right_fit, ploty, Minv):

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original, undistorted image
    original_and_shade = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    # Plot the Result
    plt.figure(figsize=(20,10))
    plt.imshow(original_and_shade)
    plt.show()

    return newwarp

def find_lanes(img):
    #print(img.dtype)
    #print(img.shape)
    img_calib = cv2.imread('./camera_cal/calibration3.jpg')
    mtx,dist,nx,ny = camera_calib(img_calib)
    left_fit = np.array([0,0,0])
    right_fit = np.array([0,0,0])
    #make a copy of the image
    original_img = np.copy(img)
    #print("Image:",original_img)
    #print("Matrix:",mtx)
    #print("Dist:",dist)
    # Ensure that the dimensions of the camera matrix are correct (3x3)
    assert mtx.shape == (3, 3), "Camera matrix dimensions are not as expected."

    # Ensure the data type of the camera matrix is compatible
    assert mtx.dtype == np.float64, "Camera matrix data type is not compatible."
    # Ensure the distortion coefficients have the correct shape (e.g., (1, 5))
    assert dist.shape == (1, 5), "Distortion coefficients have incorrect shape."

    # apply a distortion correction using camera matrix (mtx) and distortion coeff (dist) as inputs
    undistorted = cv2.undistort(original_img, mtx, dist, None, mtx)
    src = np.float32([(701,459),  #top right
                (1055,680), #bottom right
                (265,680),  #bottom left
                (580,459)]) #top left

    offset = 300 #Offset from corners for dst points.
    #print(img_size)
    dst = np.float32([(img_size[0] - offset,0), #top right
                  (img_size[0]-offset, img_size[1]), #bottom right
                   (offset,img_size[1]), #bottom left
                   (offset,0)]) #top left
    # apply a perspective transform
    M, Minv,warped = warp(undistorted,mtx,dist,src,dst)

    # combine the results of each threshold function & HLS colorspace
    combined_binary = combined_thresh(warped)
    
    # Look for lane indices
    if right_fit.any() ==0 | left_fit.any() ==0:
        # if initial values unchanged or polynomial ceases to exist use histogram to find new lane indices
        out_img, left_fitx, right_fitx, ploty, left_fit, right_fit = histogram_filter(combined_binary)
    
    else:
        # If polynomial coefficiant exist from prvious run, search around them using a refined search area 
        out_img, ploty, left_fit, right_fit, left_fitx, right_fitx = search_from_prior(combined_binary, left_fit, right_fit)
        
    # draw green shade on detected lane and warp image back to original format   
    newwarp = draw_shade(img, combined_binary, left_fit, right_fit, ploty, Minv)
    
    # Combine the result with the original, undistorted image
    result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
    
    # define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    lane_centre = (left_fitx[-1] + right_fitx[-1])/2
    centre_offset_pixels = img_size[0]/2 - lane_centre
    # convert to metres from pixels using conversion
    centre_offset_metres = xm_per_pix*centre_offset_pixels
    
    # generate data points for left and right curverad
    ploty, left_fit_cr, right_fit_cr = generate_data(ploty, left_fitx, right_fitx, ym_per_pix, xm_per_pix)
    
    # define y-value where we want radius of curvature from the bottom of the image
    y_eval = np.max(ploty)
    
    ##### Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = ((1+(2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5)//(2*abs(left_fit_cr[0]))  ## Implement the calculation of the left line here
    right_curverad = ((1+(2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5)//(2*abs(right_fit_cr[0]))  ## Implement the calculation of the right line here
    average_curvature = (left_curverad + right_curverad)/2*.001
    
    Curve_Radius = "Radius of Curvature - Centre of Path: " + str(float("%.2f" % average_curvature)) + " km"
    
    Offset = "Vehicle Position with Respect to Centre of Lane: "  + str(float("%.2f" % centre_offset_metres)) + " m"
    
    fontScale=1
    thickness=2
    fontFace = cv2.FONT_HERSHEY_SIMPLEX

    # Using CV2 putText to write text into images
    
    cv2.putText(result, Curve_Radius, (50,40), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)
    cv2.putText(result, Offset, (50,80), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)
    
    return result


if __name__ == '__main__':
    #Function to calibrate camera. 
    img_calib = cv2.imread('./camera_cal/calibration3.jpg')
    #Getting matrix and distortion coefficients to undistort image
    mtx,dist,nx,ny = camera_calib(img_calib)

    ############ USE THESE LINES OF CODE TO TEST ON IMAGES##################################
    undist_image = distort_correction(img_calib,mtx,dist)
    BGR_img = cv2.imread("./test_images/test5.jpg")
    #Convert cv2 back to RGB after reading in
    img = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2RGB)
    img_size = (img.shape[1], img.shape[0]) # width x height
    src = np.float32([(701,459),  #top right
                (1055,680), #bottom right
                (265,680),  #bottom left
                (580,459)]) #top left

    #Offset from corners for dst points.
    offset = 300 
    
    dst = np.float32([(img_size[0] - offset,0), #top right
                  (img_size[0]-offset, img_size[1]), #bottom right
                   (offset,img_size[1]), #bottom left
                   (offset,0)]) #top left
    #Perspective transform to get a top view of the lane lines
    M,Minv,warped=warp(img,mtx,dist,src,dst)
    #Thresholding values to apply sobel operator
    thresh_min = 20
    thresh_max = 200
    #SOBEL
    sobel_threshold(warped,'x',thresh_min,thresh_max)
    #Magnitude of derivative thresholding
    mag_der_thresh(warped,(30,100))
    #Direction derivative thresholding
    dir_der_thresh(warped, (1.3,1.6))
    #HLS color space thresholding
    hls_thresholding(warped, (120,255))
    #HSV color space thresholding
    hsv_thresholding(warped,(200,255))
    #COmbined thresholding
    color_combined = combined_thresh(warped)
    #Histogram filter and sliding window approach to find the lane lines
    output_image, left_fitx,right_fitx,ploty, left_fit, right_fit = histogram_filter(color_combined)
    #Using search from prior method to efficiently detect lane lines
    out,ploty, left_fitx,right_fitx,left_fit,right_fit =search_from_prior(color_combined, left_fit, right_fit)
    #Measuring curvature of the road
    curvature(ploty,left_fitx,right_fitx)
    #Plotting the lane on actual image
    draw_shade(img, color_combined, left_fit,right_fit,ploty,Minv)
    ################################# END OF SECTION FOR TESTING ON IMAGES#######################################

    ################################ USE THIS SECTION T TEST ON VIDEOS##########################################
    output = './output/output1.mp4'
    clip1 = VideoFileClip("project_video.mp4")#.subclip(26,35)
    #Function to test on videos
    test_clip = clip1.fl_image(find_lanes) #NOTE: this function expects color images!!
    test_clip.write_videofile(output, audio=False)