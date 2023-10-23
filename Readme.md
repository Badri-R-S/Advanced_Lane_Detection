# Advanced Lane Finding

Methods like Canny edge detection and hough lines were used to detect lane lines. But these methods fail to perform under factors such as lighting differnces, shadows, lane color changes and curved lanes. Hence, a more sophisticated method to detect lane lines, using traditional computer vision techniques have been implemented in this project.

## Procedure
The steps followed in this project are as follows.
- Camera calibration
- Distortion correction
- Perspective transform
- Gradient magnitude and direction thresholding
- Color thresholding
- Combined thresholding
- Detecting lane pixels and fit to find the lane boundary
- Determine the curvature of the road
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Camera calibration
Camera parameters include intrinsics, extrinsics (camera matrix 'mtx'), and distortion coefficients ('dist'). These correspondences can be calculated using multiple images of a calibration pattern, such as a checkerboard. `cv2.calibrateCamera()` has been used to determine these coefficients.
<p align="center">
<img src="https://github.com/Badri-R-S/Advanced_Lane_Detection/blob/master/output/camera_calib.jpg"
</p>

## Distortion Correction
Distortion correction needs to be done to correct image distortion utilizing our camera matrix and distortion coefficients. Distortion can 
- change the apparent size of an object in an image.
- change the apparent shape of an object in an image.
- causes the appearance of an object to change depending on where it is in the field of view.
- and make objects appear closer or farther away than they actually are.
`cv2.undistort()` has been used to achieve this.
<p align="center">
<img src="https://github.com/Badri-R-S/Advanced_Lane_Detection/blob/master/output/Undistorted_image.png"
</p>

## Perspective Transform
Transforming an image to effectively view objects from a different angle or direction. A top-down view is useful to ultimately measure the curvature of a lane line. `cv2.warpPerspective()` has been used to transform the image.
<p align="center">
<img src="https://github.com/Badri-R-S/Advanced_Lane_Detection/blob/master/output/Warped.png"
</p>

## Thresholding
In digital image processing, thresholding is the simplest method of segmenting images. From a grayscale image, thresholding can be used to create binary images.

### Sobel Operator 
The Sobel operator is at the heart of the Canny edge detection algorithm. Applying the Sobel operator to an image is a way of taking the derivative of the image in the x or y direction. As lane lines are mostly vertical, taking the derivative in the horizontal direction can provide valuable pieces of information about lane lines. `cv2.Sobel()` is used to take the derivative.
<p align="center">
<img src="https://github.com/Badri-R-S/Advanced_Lane_Detection/blob/master/output/Sobelx.png"
</p>

### Magnitude of Gradient
Apply a threshold to the overall magnitude of the gradient, in both x and y directions by taking the square root of Sobelx^2 and Sobely^2, `np.sqrt(sobelx**2 + sobely**2)` converting again to grayscale and 8-bit and a binary threshold to select pixels based on gradient strength from 20 to 100. The result was as follows:
<p align="center">
<img src="https://github.com/Badri-R-S/Advanced_Lane_Detection/blob/master/output/magnitude_of_gradient.png"
</p>

### Direction of Gradient
With lane lines, knowing the direction of the gradient can be useful, as we are interested are of a particular orientation. We can determine the direction, or orientation, of the gradient, by computing the inverse tangent (arctan) of the y gradient divided by the x gradient, arctan(sobely/sobelx) np.arctan2 `(np.absolute(sobely), np.absolute(sobelx))`
<p align="center">
<img src="https://github.com/Badri-R-S/Advanced_Lane_Detection/blob/master/output/gradient_direction.png"
</p>

### Color Thresholding
When images are converted to grayscale, to perform all the previously mentioned methods, a lot of information is lost in the color of the image. For example, yellow lane lines, the shadows in the images are valuable information, that, when used properly, can help detect lane lines exceptionally well. Hence, it is common practice to convert images to other color spaces instead of RGB to extract information.
HLS (Hue-Lightness-Saturation) and HSV (Hue-Saturation-Value) are two such color spaces.
- Hue is a value that represents color independent of any change in brightness.
- Lightness and Value represent different ways to measure the relative lightness or darkness of a color.
- Saturation is a measurement of colorfulness, i.e. as colors get lighter and closer to white, they have a lower saturation value.

The L channel represents the luminance information and is particularly good at separating the lightness of colors from their chromatic information. 
<p align="center">
<img src="https://github.com/Badri-R-S/Advanced_Lane_Detection/blob/master/output/L_channel.png"
</p>

S channel is useful for detecting and isolating regions with strong color or color contrast.
<p align="center">
<img src="https://github.com/Badri-R-S/Advanced_Lane_Detection/blob/master/output/S_channel.png"
</p>

V channel is good for detecting changes in brightness 
<p align="center">
<img src="https://github.com/Badri-R-S/Advanced_Lane_Detection/blob/master/output/V_channel.png"
</p>

### Combined thresholding
As can be seen above, the output of each thresholding method provided different information. These images can be combined and can be efficiently used to detect lane lines.
<p align="center">
<img src="https://github.com/Badri-R-S/Advanced_Lane_Detection/blob/master/output/Combined_thresholding.png"
</p>

### Histogram filter and Sliding Window approach to detect lane lines
The output from combined thresholding can be used to detect lane lines. It can be done by identifying the `Peaks` in the histogram of the image.
The image can be split into two halves and the peak for each image can be found, for the left lane and right lane.
<p align="center">
<img src="https://github.com/Badri-R-S/Advanced_Lane_Detection/blob/master/output/histogram.png"
</p>

Now that the lane lines' position have been identified, a sliding window approach can be used to draw windows in the vertical direction around the pixels that make up the lane lines.
<p align="center">
<img src="https://github.com/Badri-R-S/Advanced_Lane_Detection/blob/master/output/sliding_window_approach.png"
</p>

### A better approach to detect lane lines:
Once we have used the sliding windows function and detected lane lines and right and left lane indices, these values can be used for the next frame in the video to search around for lane lines based on activated x-values (pixels making up the lane lines) within a +/- margin of our polynomial function that makes up the lane line. This should speed up processing time as the search area has been greatly narrowed.
<p align="center">
<img src="https://github.com/Badri-R-S/Advanced_Lane_Detection/blob/master/output/serch_from_prior.png"
</p>

### Curvature and vehicle position with respect to the center
Utilizing output values in pixel space based on the polynomials calculated, a few mathematical operators can be used to determine the left and right lane curvature and average curvature based on those two values. Secondly, if we assume the camera is mounted directly in the center of the vehicle, determining the center of the lane we can calculate a vehicle offset from the centreline of the calculated lane.

## Output
Once  lane lines, curvatures, and position of the vehicle are determined, we can transpose these values onto the image. However, the image is still warped. Thus  `Minv`, the inverse transform matrix, can be used to place our lane boundaries back onto the original image.
<p align="center">
<img src="https://github.com/Badri-R-S/Advanced_Lane_Detection/blob/master/output/Final.png"
</p>

For video inputs, the same pipeline has been followed and the output is as follows:
<p align="center">
<img src="https://github.com/Badri-R-S/Advanced_Lane_Detection/blob/master/output/video.gif"
</p>

Link to the full video: https://drive.google.com/file/d/1T_RyuKg58ZjlHD7lnuOMErl_bvc7LzMN/view?usp=share_link
