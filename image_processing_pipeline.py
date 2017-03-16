import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
from tools_drawing import *
from tools_threshold_histogram import *
from scipy import signal

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "output_images/wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

prev_left_coeffs = None
prev_right_coeffs = None

def image_processing(file, filepath=False, DEBUG = False):
    
    # save the previous coeffients for tracking
    global prev_left_coeffs
    global prev_right_coeffs
    
#    plt.clf()
    
    if filepath == True:
        # Read in image
        img = cv2.imread(file)
    else:
        img = file

    # Get image shape
    imshape = img.shape
    
    # parameters to transform image from original to perspective image and vice versa
    src = np.float32(
        [[120, 720],
         [550, 470],
         [700, 470],
         [1160, 720]])

    dst = np.float32(
        [[200,720],
         [200,0],
         [1080,0],
         [1080,720]])

    # Get perspective parameters
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Parameters for greating image video
    height = img.shape[0]
    offset = 50
    offset_height = height - offset
    half_frame = img.shape[1] // 2
    steps = 6
    pixels_per_step = offset_height / steps
    window_radius = 200
    medianfilt_kernel_size = 51

    blank_canvas = np.zeros((720, 1280))
    colour_canvas = cv2.cvtColor(blank_canvas.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Generate undistored image using camera calibration parameters  
    image = cv2.undistort(img, mtx, dist, None, mtx)

    # Genrate the threshold parameters 
    have_fit = False
    curvature_checked = False
    
    xgrad_thresh_temp = (40,100)
    s_thresh_temp=(150,255)
    combined_binary = apply_threshold(image, xgrad_thresh=xgrad_thresh_temp, s_thresh=s_thresh_temp)

    # Generate perspective image for track lanes
    warped = cv2.warpPerspective(combined_binary, M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
    # Perform histogram based line tracking
    leftx, lefty, rightx, righty = histogram_pixels_v2(warped, horizontal_offset=40)    
    
    # fit polynomial to teh detected lanes 
    left_fit, left_coeffs = fit_second_order_poly(lefty, leftx, return_coeffs=True)
    right_fit, right_coeffs = fit_second_order_poly(righty, rightx, return_coeffs=True)
    if DEBUG:
        print("Left coeffs:", left_coeffs)
        print("Right coeffs: ", right_coeffs)
        # Plot data
        plt.plot(left_fit, lefty, color='green', linewidth=3)
        plt.plot(right_fit, righty, color='green', linewidth=3)
        plt.imshow(warped, cmap="gray")
        

    # Determine curvature of the lane
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = 500
    left_curverad = np.absolute(((1 + (2 * left_coeffs[0] * y_eval + left_coeffs[1])**2) ** 1.5) \
                    /(2 * left_coeffs[0]))
    right_curverad = np.absolute(((1 + (2 * right_coeffs[0] * y_eval + right_coeffs[1]) ** 2) ** 1.5) \
                     /(2 * right_coeffs[0]))
    curvature = (left_curverad + right_curverad) / 2
    min_curverad = min(left_curverad, right_curverad)
    if DEBUG:
        print("Left lane curve radius: ", left_curverad)
        print("Right lane curve radius: ", right_curverad)
    

    # TODO: if plausible parallel, continue. Else don't make `curvature_checked` = True
    if not plausible_curvature(left_curverad, right_curverad) or \
        not plausible_continuation_of_traces(left_coeffs, right_coeffs, prev_left_coeffs, prev_right_coeffs):
            if prev_left_coeffs is not None and prev_right_coeffs is not None:
                left_coeffs = prev_left_coeffs
                right_coeffs = prev_right_coeffs

    prev_left_coeffs = left_coeffs
    prev_right_coeffs = right_coeffs
    
    # Det vehicle position wrt centre
    centre = center(719, left_coeffs, right_coeffs)
        
    # Draw lane boundries on th original image
    polyfit_left = draw_poly(blank_canvas, lane_poly, left_coeffs, 30)
    polyfit_drawn = draw_poly(polyfit_left, lane_poly, right_coeffs, 30)
    if DEBUG:
        plt.imshow(polyfit_drawn, cmap="gray")
        plt.imshow(warped)

    # Convert to colour and highlight lane line area
    trace = colour_canvas
    trace[polyfit_drawn > 1] = [0,0,255]
    area = highlight_lane_line_area(blank_canvas, left_coeffs, right_coeffs)
    trace[area == 1] = [0,255,0]
    lane_lines = cv2.warpPerspective(trace, Minv, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
    if DEBUG:
        print("polyfit shape: ", polyfit_drawn.shape)
        plt.imshow(trace)
    
    

    combined_img = cv2.add(lane_lines, image)
    add_figures_to_image(combined_img, curvature=curvature, 
                         vehicle_position=centre, 
                         min_curvature=min_curverad,
                         left_coeffs=left_coeffs,
                         right_coeffs=right_coeffs)
    
    if DEBUG:
        plt.imshow(combined_img)
    
    return combined_img

# combined_img = image_pipeline("test_images/test1.jpg", filepath=True)
