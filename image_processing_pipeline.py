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

avg_left_coeffs = None
avg_right_coeffs = None
fitcount = 0

left_coeffs = None
right_coeffs = None

left_fit = None
left_fit = None

left_curverad = None
right_curverad = None
# algoUsed = None

def image_processing_pipeline(file, filepath=False, DEBUG = False):
    
    # save the previous coeffients for tracking
    global prev_left_coeffs
    global prev_right_coeffs
    
    global avg_left_coeffs
    global avg_right_coeffs
    global fitcount
    
    global left_coeffs
    global right_coeffs
    
    global left_fit
    global right_fit
    
    global left_curverad
    global right_curverad
    
    algoUsed = 'UNKNOWN'

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
    
#     ### if lane is valid then use the previous coefficients
#     if prev_info:
#         print('use previous info')
#         left_coeffs, right_coeffs, left_fit, right_fit = \
#         Utilize_previous_coeffs(warped, prev_left_coeffs, prev_right_coeffs, 100)
    
    
    ############# Logic which utilizes previous coefficients if they are valid else recompute the coefficients
    prev_info = True
    # If the fit is not valid, use windowing techinque which is in histogram_pixel function
#     print('left coeffs: ', left_coeffs)
    if left_coeffs is None or not valid_fit(left_fit, right_fit, left_curverad, right_curverad):
        prev_info = False
        print('do fit again')
        fitcount += 1
        left_coeffs, right_coeffs, left_fit, right_fit, left_curverad, right_curverad = histogram_pixel(warped)
        algoUsed = 'sliding window'
#         print ('leftfit: ', left_fit)
    else:
        print('use previous info')
        left_coeffs, right_coeffs, left_fit, right_fit, left_curverad, right_curverad = \
        Utilize_previous_coeffs(warped, prev_left_coeffs, prev_right_coeffs, 100)
        algoUsed = 'previous coefficients'
    
    # Store fit for next prediction
    if prev_info or valid_fit(left_fit, right_fit, left_curverad, right_curverad):
#         print('populating current left and right coeff to previous coeff')
        prev_left_coeffs, prev_right_coeffs = left_coeffs, right_coeffs
    else:
        print('populating current left and right coeff from previous coeff')
        left_coeffs, right_coeffs = prev_left_coeffs, prev_right_coeffs
        
    
#     ## avg coeff:
#     np.zeros((720, ))
#     if avg_left_coeffs is None:# or avg_left_coeffs < 5:
#         avg_left_coeffs.append(left_coeffs)
#         avg_right_coeffs.append(right_coeffs)
#     elif len(avg_left_coeffs) == 5:
        
#         left_coeffs = np.mean(avg_left_coeffs)
    
    
#     # Perform histogram based line tracking
# #     leftx, lefty, rightx, righty = histogram_pixels_v2(warped, horizontal_offset=40)
    
    lefty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    righty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    

    # fit polynomial to teh detected lanes 
    if DEBUG:
        print("Left coeffs:", left_coeffs)
        print("Right coeffs: ", right_coeffs)
        print("Left_fit shape: ",  left_fit.shape)
        # Plot data
        plt.plot(left_fit, lefty, color='green', linewidth=3)
        plt.plot(right_fit, righty, color='green', linewidth=3)
        plt.imshow(warped, cmap="gray")
        

#     # Determine curvature of the lane
#     # Define y-value where we want radius of curvature
#     # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = 700
#     left_curverad = np.absolute(((1 + (2 * left_coeffs[0] * y_eval + left_coeffs[1])**2) ** 1.5) \
#                     /(2 * left_coeffs[0]))
#     right_curverad = np.absolute(((1 + (2 * right_coeffs[0] * y_eval + right_coeffs[1]) ** 2) ** 1.5) \
#                      /(2 * right_coeffs[0]))
    
#     print('left curved', left_curverad)
#     print('right curved', right_curverad)
    # covert pixel to meters
    ym_per_pix = 30/720
    xm_per_pix = 3.7/700
    
    left_fit_cr = np.polyfit(lefty*ym_per_pix, left_fit*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, right_fit*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curveradm = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curveradm = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    curvature = (left_curveradm + right_curveradm) / 2
    min_curverad = curvature#min(left_curveradm, right_curveradm)
    if DEBUG:
        print("Left lane curve radius: ", left_curveradm)
        print("Right lane curve radius: ", right_curveradm)
        
    # Det vehicle position wrt center of lane
    lane_center = (right_fit[-1] + left_fit[-1])/2
    car_pos = image.shape[1] // 2
    centre = xm_per_pix* abs(lane_center - car_pos)
    if DEBUG:
        print("car position: ", car_pos, " lane center: ", lane_center,  " car position from center: ", centre, " m")
        
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
    if algoUsed == 'sliding window':
        trace[area == 1] = [255,110,0]
#     lane_lines = cv2.warpPerspective(trace, Minv, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
    else:
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
                         right_coeffs=right_coeffs,
                         algoUsed=algoUsed)
    
    if DEBUG:
        plt.imshow(combined_img)
        plt.show()
    return combined_img

# combined_img = image_processing_pipeline("test_images/test1.jpg", filepath=True, DEBUG=False)

