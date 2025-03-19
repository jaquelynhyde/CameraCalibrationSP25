# -*- coding: utf-8 -*-

# note to self: is it possible to adjust the corners we're checking against
# by a factor determined by the distortion equation the camera matrix gives?

# todo: refactor this and make some functions already -_-

import numpy as np
import cv2 as cv   
import matplotlib.pyplot as plt

debug = True

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
    
    
    # todo: investigate how to eliminate redundant calculations? 
    # visualize this stage to validate the process
    
    # reimplement this so you can look at y distances within some column
    for r in range(board_rows):
        for c in range(board_cols):
            
            if c < board_cols - 1:
                # this funny transformation is used because corners2 is a 1D array of 2D points 
                
                # maybe we calculate vertical and horizontal differences here too
                
                # this one finds the next corner to the left, assuming a rectangular target
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
    
    #todo: this step should be visualized also
    
    # now, let's measure between various distances on the chessboard with the
    # calculated ratio versus ground truth assuming the square side length is accurate
    cumulative_error_mm = 0
    cumulative_error_mm_x = 0 
    cumulative_error_mm_y = 0 # we should make a function that gets this from the arrays later
    errors = [] # we can calculate the anticipated size of these ahead of time and save a little performance time if its importance
    errors_x = []
    errors_y = []
    delerrors = []
    delerrors_x = []
    delerrors_y = []
    mean_error = 0
    
    # unclear if these need to be defined seperately for 2d histograms or if we can just use the existing arrays
    # im not really familiar with charting 2d histograms, and with the weird way the loop has to traverse the data,
    # im feeling it will take some thinking
    
    # it would also be interesting to see if the real distances calculated here and the pixel distances
    # calculated later with imgpoints2 line up
    

    
    # loop through the bottom row of columns
    # measure the distance between it and points in that column above it
    
    # modify this so you compare the bottom row against itself
    
    for c1 in range(board_cols):
        
        p1_real = [ (c1 + 1) * square_side_length , 
                    board_rows * square_side_length ]
        
        p1_pixel = corners2[(board_rows * board_cols - 1) - (c1 * board_rows)][0]
        
        if debug:
            print("for p1 : " + str(p1_real) + " // " + str(p1_pixel))
            
        for r2 in range(board_rows):
            
            p2_real = [ (c1 + 1) * square_side_length , 
                        (board_rows - r2) * square_side_length ]
            
            p2_pixel = corners2[(board_rows * board_cols - 1) - (c1 * board_rows) - r2][0]
            
            if debug:
                print("for p2 : " + str(p2_real) + " // " + str(p2_pixel))
                
            real_distance = np.sqrt( (p2_real[0] - p1_real[0]) ** 2 + (p2_real[1] - p1_real[1]) ** 2 )
            
            # distance_y has the opposite order so that the sign is correct
            real_distance_x = p2_real[0] - p1_real[0]
            real_distance_y = p1_real[1] - p2_real[1]
            
            pixel_distance = np.sqrt( (p2_pixel[0] - p1_pixel[0]) ** 2 + (p2_pixel[1] - p1_pixel[1]) ** 2 )
            
            pixel_distance_x = p2_pixel[0] - p1_pixel[0]
            pixel_distance_y = p1_pixel[1] - p2_pixel[1]
            
            theo_distance = pixel_distance * mm_pixel_ratio
            theo_distance_x = pixel_distance_x * mm_pixel_ratio 
            theo_distance_y = pixel_distance_y * mm_pixel_ratio
            
            errors.append(theo_distance - real_distance)
            errors_x.append(theo_distance_x - real_distance_x)
            errors_y.append(theo_distance_y - real_distance_y)
            
            
            
            
            """
            for p2 : [468, 132] // [1148.6586   482.43848]
            real distance : 48.0 | pixel_distance : 104.22389737663727 | theoretical real distance : 46.47958926741875
            error : -1.5204107325812473 | error_x : 1.0076540330778292 | error_y : 1.531334731697207
            delta error : 0.4575718061970804 | delta error_x : -0.021060840305077537 | delta error_y : -0.45757890518490285
            for p2 : [468, 120] // [1149.1991   456.40964]
            real distance : 60.0 | pixel_distance : 130.2583395362692 | theoretical real distance : 58.08988411192253
            error : -1.9101158880774705 | error_x : 1.2487070886406941 | error_y : 1.9235386172386626
            delta error : -4.303620172049829 | delta error_x : -0.01439894606964267 | delta error_y : 4.303643518000001
            """
            
            # double check the indexing here later, sometimes the delta is too big? 
            if (c1 + r2) > 0:
                delerrors.append(errors[c1 + r2] - errors[c1 + r2 - 1])
                delerrors_x.append(errors_x[c1 + r2] - errors_x[c1 + r2 - 1])
                delerrors_y.append(errors_y[c1 + r2] - errors_y[c1 + r2 - 1])
            else:
                delerrors.append(0)
                delerrors_x.append(0)
                delerrors_y.append(0)
                
            
            if debug:
                print("real distance : " + str(real_distance) + " | " + "pixel_distance : " + str(pixel_distance)  + " | " + "theoretical real distance : " + str(theo_distance) )
                print("real distance x: " + str(real_distance_x) + " | " + "pixel_distance x: " + str(pixel_distance_x)  + " | " + "theoretical real distance x: " + str(theo_distance_x) )
                print("real distance y: " + str(real_distance_y) + " | " + "pixel_distance y: " + str(pixel_distance_y)  + " | " + "theoretical real distance y: " + str(theo_distance_y) )
                print("error : " + str(theo_distance - real_distance) + " | " + "error_x : " + str(theo_distance_x - real_distance_x)  + " | " + "error_y : " + str(theo_distance_y - real_distance_y) )
                if (c1 + r2) > 0:
                    print("delta error : " + str(errors[c1 + r2] - errors[c1 + r2 - 1]) + " | " + "delta error_x : " + str(errors_x[c1 + r2] - errors_x[c1 + r2 - 1])  + " | " + "delta error_y : " + str(errors_y[c1 + r2] - errors_y[c1 + r2 - 1]) )
 
    # really important to do:
        # retrieve all the x and y points associated with the error calculation array
        # plot them here
        
        # work on your poster presentation
        # https://research.mnsu.edu/undergraduate-research-center/undergraduate-research-center-present-and-publish/undergraduate-research-symposium/
 
       
    abs_errors = np.abs(errors)
    abs_errors_x = np.abs(errors_x)
    abs_errors_y = np.abs(errors_y)
    
    plt.figure(dpi=300)
    plt.imshow(gray_image)
    
    # should this part be using the real world distances, or pixel locations, or...?
    x_array = np.arange(square_side_length, square_side_length * (board_cols + 1), square_side_length)
    y_array = np.arange(square_side_length, square_side_length * (board_rows + 1), square_side_length)
    
    print("real x coords")
    print(x_array)
    print("real y coords")
    print(y_array)
        
    x_arranged, y_arranged = np.meshgrid(x_array, y_array)
    
    print("x_arranged")
    print(x_arranged)
    print("y_arranged")
    print(y_arranged)
    
    # make a version of this that uses del-errors
    
    e_array = np.array(errors)
    x_e_array = np.array(errors_x)
    y_e_array = np.array(errors_y)
    
    d_e_array = np.array(delerrors)
    x_d_e_array = np.array(delerrors_x)
    y_d_e_array = np.array(delerrors_y)
        
    e_arranged = e_array.reshape((board_cols,board_rows))    
    x_e_arranged = x_e_array.reshape((board_cols,board_rows)) 
    y_e_arranged = y_e_array.reshape((board_cols,board_rows))    
    
    d_e_arranged = d_e_array.reshape((board_cols,board_rows))    
    x_d_e_arranged = x_d_e_array.reshape((board_cols,board_rows)) 
    y_d_e_arranged = y_d_e_array.reshape((board_cols,board_rows))    
    
    print("errors")
    print(errors)
    print("e_array")
    print(e_array)
    print("e_arranged")
    print(e_arranged)
    
    e_arranged = np.rot90(e_arranged)
    
    d_e_arranged = np.rot90(d_e_arranged)
    
    print("rot90 e_arranged")
    print(e_arranged)
    
    e_arranged = np.flip(e_arranged)
    e_arranged = np.fliplr(e_arranged)
    
    d_e_arranged = np.flip(d_e_arranged)
    d_e_arranged = np.fliplr(d_e_arranged)
    
    print("flip e_arranged")
    print(e_arranged)
    
    print('x_e_arranged')
    print(x_e_arranged)
    x_e_arranged = np.rot90(x_e_arranged)
    print('rot x_e_arranged')
    print(x_e_arranged)
    x_e_arranged = np.flip(x_e_arranged)
    print('flip x_e_arranged')
    print(x_e_arranged)
    x_e_arranged = np.fliplr(x_e_arranged)
    print('fliplr x_e_arranged')
    print(x_e_arranged)
    
    x_d_e_arranged = np.rot90(x_d_e_arranged)
    x_d_e_arranged = np.flip(x_d_e_arranged)
    x_d_e_arranged = np.fliplr(x_d_e_arranged)
    
    print('y_e_arranged')
    print(y_e_arranged)
    y_e_arranged = np.rot90(y_e_arranged)
    print('rot x_e_arranged')
    print(x_e_arranged)
    y_e_arranged = np.flip(y_e_arranged)
    print('flip x_e_arranged')
    print(x_e_arranged)
    y_e_arranged = np.fliplr(y_e_arranged)
    print('flip lrx_e_arranged')
    print(x_e_arranged)
    
    y_d_e_arranged = np.rot90(y_d_e_arranged)
    y_d_e_arranged = np.flip(y_d_e_arranged)
    y_d_e_arranged = np.fliplr(y_d_e_arranged)
    
    # check out the example below and figure out how to reshape errors so its correct
    
    # todo: learn more abt subplots
    # make more of these that show x and y error once u know the syntax is right
    contourfig, ax2 = plt.subplots(layout = 'constrained')
    CS = ax2.contourf(x_arranged, y_arranged, e_arranged, levels = 25, cmap = 'inferno')
    CS2 = ax2.contour(CS, levels = CS.levels[::2], colors='b')
    ax2.set_title('Error in Calculated Real Distance from Bottom')
    ax2.set_xlabel('?')
    ax2.set_ylabel('?') # not sure what to label these
    cbar = contourfig.colorbar(CS)
    cbar.ax.set_ylabel('Error Magnitude')
    cbar.add_lines(CS2)
    
    plt.show()

    xcontourfig, xax2 = plt.subplots(layout = 'constrained')
    xCS = xax2.contourf(x_arranged, y_arranged, x_e_arranged, levels = 25, cmap = 'inferno')
    xCS2 = xax2.contour(xCS, levels = xCS.levels[::2], colors='b')
    xax2.set_title('Error in Calculated X Real Distance from Bottom')
    xax2.set_xlabel('?')
    xax2.set_ylabel('?') # not sure what to label these
    xcbar = xcontourfig.colorbar(xCS)
    xcbar.ax.set_ylabel('Error Magnitude')
    xcbar.add_lines(xCS2)
    
    plt.show()
    
    ycontourfig, yax2 = plt.subplots(layout = 'constrained')
    yCS = yax2.contourf(x_arranged, y_arranged, y_e_arranged, levels = 25, cmap = 'inferno')
    yCS2 = yax2.contour(yCS, levels = yCS.levels[::2], colors='b')
    yax2.set_title('Error in Calculated Y Real Distance from Bottom')
    yax2.set_xlabel('?')
    yax2.set_ylabel('?') # not sure what to label these
    ycbar = ycontourfig.colorbar(yCS)
    ycbar.ax.set_ylabel('Error Magnitude')
    ycbar.add_lines(yCS2)
    
    plt.show()
    
    dcontourfig, dax2 = plt.subplots(layout = 'constrained')
    dCS = dax2.contourf(x_arranged, y_arranged, d_e_arranged, levels = 5, cmap = 'inferno')
    dCS2 = dax2.contour(dCS, levels = dCS.levels[::2], colors='b')
    dax2.set_title('Delta Error in Calculated Real Distance from Bottom')
    dax2.set_xlabel('?')
    dax2.set_ylabel('?') # not sure what to label these
    dcbar = dcontourfig.colorbar(dCS)
    dcbar.ax.set_ylabel('Error Magnitude')
    dcbar.add_lines(dCS2)
    
    plt.show()

    xdcontourfig, xdax2 = plt.subplots(layout = 'constrained')
    xdCS = xdax2.contourf(x_arranged, y_arranged, x_d_e_arranged, levels = 5, cmap = 'inferno')
    xdCS2 = xdax2.contour(xdCS, levels = xdCS.levels[::2], colors='b')
    xdax2.set_title('Error in Calculated X Real Distance from Bottom')
    xdax2.set_xlabel('?')
    xdax2.set_ylabel('?') # not sure what to label these
    xdcbar = xdcontourfig.colorbar(xdCS)
    xdcbar.ax.set_ylabel('Error Magnitude')
    xdcbar.add_lines(xdCS2)
    
    plt.show()
    
    ydcontourfig, ydax2 = plt.subplots(layout = 'constrained')
    ydCS = ydax2.contourf(x_arranged, y_arranged, y_d_e_arranged, levels = 5, cmap = 'inferno')
    ydCS2 = ydax2.contour(ydCS, levels = ydCS.levels[::2], colors='b')
    ydax2.set_title('Error in Calculated Y Real Distance from Bottom')
    ydax2.set_xlabel('?')
    ydax2.set_ylabel('?') # not sure what to label these
    ydcbar = ydcontourfig.colorbar(ydCS)
    ydcbar.ax.set_ylabel('Error Magnitude')
    ydcbar.add_lines(ydCS2)
    
    plt.show()
    
    # now let's graph these a different way and overlay them on the image
    
    # first we need to build a new x array and y array based on the corners in corners2 
    
    x_px_array = np.zeros((board_rows,board_cols))
    y_px_array = np.zeros((board_rows,board_cols))

    print('x/y px array zeroes')
    print(x_px_array)

    for row in range(board_rows):
        for col in range(board_cols):
            x_px_array[row, col] = corners2[board_rows * board_cols - 1 - row - (col * board_rows)][0][0]
            y_px_array[row, col] = corners2[board_rows * board_cols - 1 - row - (col * board_rows)][0][1]
    
    print('x px array filled')
    print(x_px_array)
    print('y px array filled')
    print(y_px_array)
            
    overlayfig, ax3 = plt.subplots(layout = 'constrained')     
    ax3.imshow(gray_image, cmap = 'gray')
    ax3.set_title('Error in Calculated Real Distance from Bottom of Target')
    overlay_contour = ax3.contourf(x_px_array, y_px_array, e_arranged, levels = 25, cmap = 'viridis', alpha = 0.7)   
    overlay_contour2 = ax3.contour(overlay_contour, levels = overlay_contour.levels[::2], colors='b')
    ocbar = overlayfig.colorbar(overlay_contour)
    ocbar.ax.set_ylabel('Error Magnitude (mm)')
    ocbar.add_lines(overlay_contour2)
    
    plt.show()
    
    xoverlayfig, xax3 = plt.subplots(layout = 'constrained')     
    xax3.imshow(gray_image, cmap = 'gray')
    xax3.set_title('Error in Calculated Real X Axis Distance from Bottom of Target')
    xoverlay_contour = xax3.contourf(x_px_array, y_px_array, x_e_arranged, levels = 25, cmap = 'viridis', alpha = 0.7)   
    xoverlay_contour2 = xax3.contour(xoverlay_contour, levels = xoverlay_contour.levels[::2], colors='b')
    xocbar = xoverlayfig.colorbar(xoverlay_contour)
    xocbar.ax.set_ylabel('Error Magnitude (mm)')
    xocbar.add_lines(xoverlay_contour2)
    
    plt.show()
    
    yoverlayfig, yax3 = plt.subplots(layout = 'constrained')     
    yax3.imshow(gray_image, cmap = 'gray')
    yax3.set_title('Error in Calculated Real Y Axis Distance from Bottom of Target')
    yoverlay_contour = yax3.contourf(x_px_array, y_px_array, y_e_arranged, levels = 25, cmap = 'viridis', alpha = 0.7)   
    yoverlay_contour2 = yax3.contour(yoverlay_contour, levels = yoverlay_contour.levels[::2], colors='b')
    yocbar = yoverlayfig.colorbar(yoverlay_contour)
    yocbar.ax.set_ylabel('Error Magnitude (mm)')
    yocbar.add_lines(overlay_contour2)
    
    plt.show()

    doverlayfig, dax3 = plt.subplots(layout = 'constrained')     
    dax3.imshow(gray_image, cmap = 'gray')
    dax3.set_title('Delta Error in Calculated Real Distance from Bottom of Target')
    doverlay_contour = dax3.contourf(x_px_array, y_px_array, d_e_arranged, levels = 5, cmap = 'viridis', alpha = 0.7)   
    doverlay_contour2 = dax3.contour(doverlay_contour, levels = doverlay_contour.levels[::2], colors='b')
    docbar = doverlayfig.colorbar(doverlay_contour)
    docbar.ax.set_ylabel('Error Magnitude (mm)')
    docbar.add_lines(doverlay_contour2)
    
    plt.show()
    
    xdoverlayfig, xdax3 = plt.subplots(layout = 'constrained')     
    xdax3.imshow(gray_image, cmap = 'gray')
    xdax3.set_title('Delta Error in Calculated Real X Axis Distance from Bottom of Target')
    xdoverlay_contour = xdax3.contourf(x_px_array, y_px_array, x_d_e_arranged, levels = 5, cmap = 'viridis', alpha = 0.7)   
    xdoverlay_contour2 = xdax3.contour(xdoverlay_contour, levels = xdoverlay_contour.levels[::2], colors='b')
    xdocbar = xdoverlayfig.colorbar(xdoverlay_contour)
    xdocbar.ax.set_ylabel('Error Magnitude (mm)')
    xdocbar.add_lines(xdoverlay_contour2)
    
    plt.show()
    
    ydoverlayfig, ydax3 = plt.subplots(layout = 'constrained')     
    ydax3.imshow(gray_image, cmap = 'gray')
    ydax3.set_title('Delta Error in Calculated Real Y Axis Distance from Bottom of Target')
    ydoverlay_contour = ydax3.contourf(x_px_array, y_px_array, y_d_e_arranged, levels = 5, cmap = 'viridis', alpha = 0.7)   
    ydoverlay_contour2 = ydax3.contour(ydoverlay_contour, levels = ydoverlay_contour.levels[::2], colors='b')
    ydocbar = ydoverlayfig.colorbar(ydoverlay_contour)
    ydocbar.ax.set_ylabel('Error Magnitude (mm)')
    ydocbar.add_lines(doverlay_contour2)
    
    plt.show()

#https://matplotlib.org/stable/gallery/images_contours_and_fields/layer_images.html
#https://matplotlib.org/stable/gallery/images_contours_and_fields/contourf_demo.html

# todo: refactor so you're using the same kind of syntax as you do w/ the contour plot above

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
    
    plt.hist(delerrors, bins = 20, edgecolor='black')
    plt.xlabel("Error (mm)")
    plt.ylabel("Frequency")
    plt.title("Total Distance Error Delta Distribution")
    plt.show()
        
    plt.hist(delerrors_x, bins = 20, edgecolor='black')
    plt.xlabel("X Error (mm)")
    plt.ylabel("Frequency")
    plt.title("Total X Distance Error Delta Distribution")
    plt.show()
    
    plt.hist(delerrors_y, bins = 20, edgecolor='black')
    plt.xlabel("Y Error (mm)")
    plt.ylabel("Frequency")
    plt.title("Total Y Distance Error Delta Distribution")
    plt.show()
    
    plt.hist(abs_errors, bins = 20, edgecolor='black')
    plt.xlabel("Error (mm)")
    plt.ylabel("Frequency")
    plt.title("Total Absolute Distance Error Distribution")
    plt.show()
        
    plt.hist(abs_errors_x, bins = 20, edgecolor='black')
    plt.xlabel("X Error (mm)")
    plt.ylabel("Frequency")
    plt.title("Total Absolute X Distance Error Distribution")
    plt.show()
    
    plt.hist(abs_errors_y, bins = 20, edgecolor='black')
    plt.xlabel("Y Error (mm)")
    plt.ylabel("Frequency")
    plt.title("Total Absolute Y Distance Error Distribution")
    plt.show()
    
    print("abs_Mean Distance Error: " + str(np.mean(abs_errors)))
    print("abs_Median Distance Error: " + str(np.median(abs_errors)))
    print("abs_Distance Error Standard Deviation: " + str(np.std(abs_errors)))
    print("abs_Distance Error Variance: " + str(np.var(abs_errors)))
    print("abs_Distance Error Percentile 1: " + str(np.percentile(abs_errors, 1)))
    print("abs_Distance Error Percentile 25: " + str(np.percentile(abs_errors, 25)))
    print("abs_Distance Error Percentile 50: " + str(np.percentile(abs_errors, 50)))
    print("abs_Distance Error Percentile 75: " + str(np.percentile(abs_errors, 75)))    
    print("abs_Distance Error Percentile 99: " + str(np.percentile(abs_errors, 99)))
    
    print("abs_X Mean Distance Error: " + str(np.mean(abs_errors_x)))
    print("abs_X Median Distance Error: " + str(np.median(abs_errors_x)))
    print("abs_X Distance Error Standard Deviation: " + str(np.std(abs_errors_x)))
    print("abs_X Distance Error Variance: " + str(np.var(abs_errors_x)))
    print("abs_X Distance Error Percentile 1: " + str(np.percentile(abs_errors_x, 1)))
    print("abs_X Distance Error Percentile 25: " + str(np.percentile(abs_errors_x, 25)))
    print("abs_X Distance Error Percentile 50: " + str(np.percentile(abs_errors_x, 50)))
    print("abs_X Distance Error Percentile 75: " + str(np.percentile(abs_errors_x, 75)))    
    print("abs_X Distance Error Percentile 99: " + str(np.percentile(abs_errors_x, 99)))
    
    print("abs_Y Mean Distance Error: " + str(np.mean(abs_errors_y)))
    print("abs_Y Median Distance Error: " + str(np.median(abs_errors_y)))
    print("abs_Y Distance Error Standard Deviation: " + str(np.std(abs_errors_y)))
    print("abs_Y Distance Error Variance: " + str(np.var(abs_errors_y)))
    print("abs_Y Distance Error Percentile 1: " + str(np.percentile(abs_errors_y, 1)))
    print("abs_Y Distance Error Percentile 25: " + str(np.percentile(abs_errors_y, 25)))
    print("abs_Y Distance Error Percentile 50: " + str(np.percentile(abs_errors_y, 50)))
    print("abs_Y Distance Error Percentile 75: " + str(np.percentile(abs_errors_y, 75)))    
    print("abs_Y Distance Error Percentile 99: " + str(np.percentile(abs_errors_y, 99)))
    
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
        
    projection_error_mean = projection_error_abs / len(objpoints[0]) # this is dividing by 1 because objpoints is a weird array -_- fix later
    
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
