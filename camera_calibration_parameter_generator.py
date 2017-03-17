

'''
Camera Calibration
'''

import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
def calibrate_camera(img_dir, nx = 8, ny = 6, DRAW = False):
    # nx = 9
    # ny = 6
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # plot figures
    if DRAW:
        plt.figure(figsize=(10,5))
        plt.axis('off')

    # Make a list of calibration images
    images = glob.glob(img_dir + '/' + 'calibration*.jpg')
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        if DRAW:
            plt.subplot(4,5,idx+1)
        ret = False
        #read image
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            if(DRAW):
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (9,6), corners, ret)
                plt.imshow(img)
                plt.axis('off')    
    # Do camera calibration given object points and image points
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    return mtx, dist

mtx, dist = calibrate_camera('camera_cal',9,6, False)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "output_images/wide_dist_pickle.p", "wb" ) )
print('pickle file saved !!!')   
