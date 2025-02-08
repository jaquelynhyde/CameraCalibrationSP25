# -*- coding: utf-8 -*-
"""

takes:
    a series of images, the first should be a front view of the calib target
    the length of a side of a square in the calibration target
    
outputs:
    an estimation of the mm/pixel ratio in the front view of the calib target
    a view of each of the images, before and after distortion
    measurements of lengths across the target, before and after distortion
"""

import numpy as np
import cv2 as cv   
import matplotlib.pyplot as plt

debug = False

window_name = "Camera Validation"

wait_time = 250

square_side_length = 12 # millimeters

board_rows = 15
board_cols = 39

image_height = 964
image_width = 1280

num_channels = 3

images = [cv.imread(cv.samples.findFile("input_01.jpg")) , 
          cv.imread(cv.samples.findFile("input_01.jpg")) , 
          cv.imread(cv.samples.findFile("input_01.jpg")) ]

# first, we are going to calibrate the camera with the location of chessboard corners in the first image

# findchessboardcorners wants a grayscale image
# you can use edge detection here as well but its unclear if it provides some benefit

# todo: take a look at other image preprocessing steps and see how they affect the error
gray_image = cv.cvtColor(images[0], cv.COLOR_BGR2GRAY)

termination_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# these are numpy arrays with the calibration target's corner's points in 3d space
# the implementation here bases the model on the assumption that the target is
# on the x,y plane where z = 0 and the camera is rotated accordingly 
objp = np.zeros((board_rows*board_cols, 3), np.float32)
print("objp zeroes")
print(objp)
objp[:,:2] = np.mgrid[0:board_rows,0:board_cols].T.reshape(-1,2) * square_side_length
print("objp populated")
print(objp)

objpoints = []
imgpoints = []

# opencv finds the points of the chessboard corners in the image
ret, corners = cv.findChessboardCorners(gray_image, (board_rows, board_cols), None)
print("ret")
print(ret)
print("corners")
print(corners)

# if we successfully found corners...
if ret == True:
    # add the object points to our collection
    objpoints.append(objp)
    print("objpoints")
    print(objpoints)
    
    # refine the locations of the points with 'sub pixels' and save them too
    corners2 = cv.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), termination_criteria)
    print("corners2")
    print(corners2)
    imgpoints.append(corners2)
    print("imgpoints")
    print(imgpoints)

    # loop through corners2 to find the average distance in pixels between 
    # adjacent squares on the chessboard
    
    # it may be good to recalculate this for different regions of the image? 
    # generate some distance-between-points-gradient? 
    
    total_horizontal_distance = 0
    total_vertical_distance = 0
    num_horizontal_distances = 0
    num_vertical_distances = 0
    
    
    for r in range(board_rows):
        for c in range(board_cols):
            
            if c < board_cols - 1:
                # this funny transformation is used because corners2 is a 1D array of 2D points 
                
                # maybe we calculate vertical and horizontal differences here too
                
                # this one finds the next corner to the right, assuming a rectangular target
                point1 = corners2[c * board_rows + r][0]        
                point2 = corners2[(c + 1) * board_rows + r][0]
                
                distance = np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
                
                total_horizontal_distance += distance
                num_horizontal_distances += 1
            if r < board_rows - 1:
                # this one finds the next corner down, assuming a rectangular target
                point1 = corners2[c * board_rows + r][0]
                point2 = corners2[c * board_rows + r + 1][0]
                
                distance = np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

                total_vertical_distance += distance
                num_vertical_distances += 1
                
    average_horizontal_distance = total_horizontal_distance / num_horizontal_distances
    average_vertical_distance = total_vertical_distance / num_vertical_distances 
    average_distance = ( average_horizontal_distance + average_vertical_distance ) / 2
    
    mm_pixel_ratio = square_side_length / average_distance

    print("avg pixel distance:")
    print(average_distance)
    print("mm pixel ratio:")
    print(mm_pixel_ratio)
    
    # now, let's measure between various distances on the chessboard with the
    # calculated ratio versus ground truth assuming the square side length is accurate
    cumulative_error_mm = 0
    cumulative_error_mm_x = 0 
    cumulative_error_mm_y = 0 # we should make a function that gets this from the arrays later
    errors = [] # we can calculate the anticipated size of these ahead of time and save a little performance time if its importance
    errors_x = []
    errors_y = []
    mean_error = 0
    
    # unclear if these need to be defined seperately for 2d histograms or if we can just use the existing arrays
    # im not really familiar with charting 2d histograms, and with the weird way the loop has to traverse the data,
    # im feeling it will take some thinking
    
    # it would also be interesting to see if the real distances calculated here and the pixel distances
    # calculated later with imgpoints2 line up
    real_x_coords = np.arange(0, square_side_length * (board_cols + 1), square_side_length)
    real_y_coords = np.arange(0, square_side_length * (board_rows + 1), square_side_length)
    
    print("real x coords")
    print(real_x_coords)
    print("real y coords")
    print(real_y_coords)
    
    image_data = np.zeros((len(real_x_coords), len(real_y_coords)))
    
    print("image_data (empty)")
    print(image_data)
    
    # five four loops with changing images (from an array)
    
    # sorry for the four for loops 😪
    for c1 in range(board_cols):
        for r1 in range(board_rows):
            
            # was having trouble directly getting these from objpoints
            # so we manually calculate them, making sure they match the form of
            # corners2 (i.e. - starting from the top right corner and going down rows in each column)
            p1_real = [ (board_cols - c1 - 1) * square_side_length ,
                        r1 * square_side_length] 
            
            p1_pixel = corners2[c1 * board_rows + r1][0]
            
            if debug:
                print("for r1, c1 " + str(r1) + ", " + str(c1))
                print("p1 real / p1_pixel " + str(p1_real) + ", " + str(p1_pixel))
            
            
            for c2 in range(board_cols):
                for r2 in range(board_rows):
                    p2_real = [ (board_cols - c2 - 1) * square_side_length ,
                                r2 * square_side_length] 
                    p2_pixel = corners2[c2 * board_rows + r2][0]
                    
                    real_distance = np.sqrt( (p2_real[0] - p1_real[0]) ** 2 + (p2_real[1] - p1_real[1]) ** 2 )
                    
                    real_distance_x = p2_real[0] - p1_real[0]
                    real_distance_y = p2_real[1] - p1_real[1]
                    
                    pixel_distance = np.sqrt( (p2_pixel[0] - p1_pixel[0]) ** 2 + (p2_pixel[1] - p1_pixel[1]) ** 2 )
                    
                    pixel_distance_x = p2_pixel[0] - p1_pixel[0]
                    pixel_distance_y = p2_pixel[1] - p1_pixel[1]
                    
                    theo_distance = pixel_distance * mm_pixel_ratio
                    theo_distance_x = pixel_distance_x * mm_pixel_ratio 
                    theo_distance_y = pixel_distance_y * mm_pixel_ratio
                    
                    errors.append(theo_distance - real_distance)
                    errors_x.append(theo_distance_x - real_distance_x)
                    errors_y.append(theo_distance_y - real_distance_y)
                    
                    if debug:
                        print("for r2, c2 " + str(r2) + ", " + str(c2))
                        print("p2 real / p2_pixel " + str(p2_real) + ", " + str(p2_pixel))
                        print("real distance : " + str(real_distance) )
                        print("pixel_distance : " + str(pixel_distance) )
                        print("theoretical real distance : " + str(theo_distance))
      
    #to do.. determine bin number in a smarter way...
    
    plt.hist(errors, bins = 20, edgecolor='black')
    plt.xlabel("Error (mm)")
    plt.ylabel("Frequency")
    plt.title("Total Distance Error Distribution")
    plt.show()
        
    plt.hist(errors_x, bins = 20, edgecolor='black')
    plt.xlabel("X Error (mm)")
    plt.ylabel("Frequency")
    plt.title("Total X Distance Error Distribution")
    plt.show()
    
    plt.hist(errors_y, bins = 20, edgecolor='black')
    plt.xlabel("Y Error (mm)")
    plt.ylabel("Frequency")
    plt.title("Total Y Distance Error Distribution")
    plt.show()
    
    print("Mean Distance Error: " + str(np.mean(errors)))
    print("Median Distance Error: " + str(np.median(errors)))
    print("Distance Error Standard Deviation: " + str(np.std(errors)))
    print("Distance Error Variance: " + str(np.var(errors)))
    print("Distance Error Percentile 1: " + str(np.percentile(errors, 1)))
    print("Distance Error Percentile 25: " + str(np.percentile(errors, 25)))
    print("Distance Error Percentile 50: " + str(np.percentile(errors, 50)))
    print("Distance Error Percentile 75: " + str(np.percentile(errors, 75)))    
    print("Distance Error Percentile 99: " + str(np.percentile(errors, 99)))
    
    print("X Mean Distance Error: " + str(np.mean(errors_x)))
    print("X Median Distance Error: " + str(np.median(errors_x)))
    print("X Distance Error Standard Deviation: " + str(np.std(errors_x)))
    print("X Distance Error Variance: " + str(np.var(errors_x)))
    print("X Distance Error Percentile 1: " + str(np.percentile(errors_x, 1)))
    print("X Distance Error Percentile 25: " + str(np.percentile(errors_x, 25)))
    print("X Distance Error Percentile 50: " + str(np.percentile(errors_x, 50)))
    print("X Distance Error Percentile 75: " + str(np.percentile(errors_x, 75)))    
    print("X Distance Error Percentile 99: " + str(np.percentile(errors_x, 99)))
    
    print("Y Mean Distance Error: " + str(np.mean(errors_y)))
    print("Y Median Distance Error: " + str(np.median(errors_y)))
    print("Y Distance Error Standard Deviation: " + str(np.std(errors_y)))
    print("Y Distance Error Variance: " + str(np.var(errors_y)))
    print("Y Distance Error Percentile 1: " + str(np.percentile(errors_y, 1)))
    print("Y Distance Error Percentile 25: " + str(np.percentile(errors_y, 25)))
    print("Y Distance Error Percentile 50: " + str(np.percentile(errors_y, 50)))
    print("Y Distance Error Percentile 75: " + str(np.percentile(errors_y, 75)))    
    print("Y Distance Error Percentile 99: " + str(np.percentile(errors_y, 99)))
    
    # it would be benefical to look at how this changes if we ONLY look at the x/y axis
    
    # also, we should graph the x/y error at each point we calculate it at - 
    
    # now, let's calibrate the camera matrix, and use it to undistort the image
    # opencv needs a set of points in the real world and in the image to do this
    # the cursed numpy addressing gives the function the resolution of the image the way it likes
    # but i'm not 100% sure on how that parameter affects the final matrix 
    # i imagine it would change the image center or focal length or something to be in terms of pixels?
    
    # i wonder if we could improve this by finding objpoints and imgpoints for a set of images 
    # where the calibration target moves along the z axis
    # or develop a calibration target with some 3d element to it
    
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray_image.shape[::-1], None, None)
    
    print("intrinsic parameter matrix:")
    print(mtx)
    print("distortion parameters:")  # check to see if the x and y distortion equations are different
    print(dist)
    print("rotation matrix:")
    print(rvecs)
    print("translation matrix:")
    print(tvecs)
    
    #calibrateCamera returns even more data such as standard deviations etc
    # read the docs here: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d 
    
    # the opencv docs provide this method of calculating error, which to my understanding,
    # goes through every point and finds the imgpoint it *should* occupy 
    
    # the fancy way they do this makes me feel like my loop could be optimized a lot
    # ( which was already obvious, but, )
    # i don't know what the norm of an array/matrix stuff does i need to research that
    
    projection_error_abs = 0 
    projection_error_mean = 0 
    
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2) # ??? 😐
        projection_error_abs += error
        
    # to do - loop through imgpoints and collect distance between points in each array
        
    projection_error_mean = projection_error_abs / len(objpoints)
    
    print("absolute projection error:")
    print(projection_error_abs)
    print("mean projection error:")
    print(projection_error_mean)
    
    # we can undistort the original image
    # (and perhaps, considering the fisheye lens, relocate image points and improve our result?)
    # (so the ratio is linear across the image?)
    # ( that would take reformating the code a little to bundle some of the above into functions and etc... )
    undistorted = cv.undistort(gray_image, mtx, dist, None)
    
    # we can also find a new camera matrix, with a new image with the same camera in a different position
    # and call undistort again, with the new image and an additional argument (the new matrix)

    # to-do: learn more about distortion coefficients and how they relate obj/img points
    # maybe see what the output is like with artificial images
    
    # to-do: 
        # create some descriptions of the central tendency of the data
        # i.e. create a histogram of all calculated errors
        # the bins should be based off value ranges as well as sign
    
cv.destroyAllWindows()
