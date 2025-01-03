# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 18:26:56 2025

@author: Jaquelyn

sourced from:
https://docs.opencv.org/4.x/d9/db7/tutorial_py_table_of_contents_calib3d.html
https://docs.opencv.org/4.x/dc/d71/tutorial_py_optimization.html

"""

import numpy as np
import cv2 as cv

class CameraCalibration:
    def __init__(self):
        self.input_image = cv.imread(cv.samples.findFile("Input_Image.jpg"))
        self.gray = None
        self.gray_color_image = None
        
        self.window_name = 'Camera Calibration'
        self.wait_time = 10
        self.min_thresh_val = 127
        self.max_thresh_val = 200
        self.cal_target_rows = 5
        self.cal_target_cols = 8
        self.grid_square_mm = 15
        self.mm_pixel_ratio = -1
        self.last_time = 0
        
    # this could probably be updated to maintain a dictionary of events to track?
    @staticmethod
    def debug_timer(start, event = '[UNTITLED]'):
        if start == True:
            print('hi')
        else:
            print('hello')
        
    def calibrate(self):
        cv.namedWindow(self.window_name)
        
        self.gray = cv.cvtColor(self.input_image, cv.COLOR_BGR2GRAY)
        
        self.gray_color_image = cv.cvtColor(self.gray, cv.COLOR_GRAY2BGR)
        
        self.edges = cv.Canny(self.gray, self.min_thresh_val, self.max_thresh_val)
        
        self.edges_color_image = cv.cvtColor(self.edges, cv.COLOR_GRAY2BGR)
        
        termination_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        objp = np.zeros((self.cal_target_rows*self.cal_target_cols,3), np.float32)
        objp[:,:2] = np.mgrid[0:self.cal_target_rows,0:self.cal_target_cols].T.reshape(-1,2)
        
        objpoints = [] # points in 3d space (real world)
        imgpoints = [] # points in 2d space (image plane)
        
        ret, corners = cv.findChessboardCorners(self.gray, (self.cal_target_rows, self.cal_target_cols), None)
        
        cv.imshow(self.window_name, self.input_image)
        
        cv.waitKey(0)
        
        cv.imshow(self.window_name, self.gray_color_image)
        
        if ret == True:
            objpoints.append(objp)
            
            corners2 = cv.cornerSubPix(self.gray, corners, (11,11), (-1,-1), termination_criteria)
            
            print('Corners Array :')
            print(corners2)
            
            imgpoints.append(corners2)
            
            total_horizontal_distance = 0
            total_vertical_distance = 0
            num_horizontal_distances = 0
            num_vertical_distances = 0
            
            for r in range(self.cal_target_rows):
                print('testing row ' + str(r))
                for c in range(self.cal_target_cols):
                    print('testing column ' + str(c))
                    
                    if c < self.cal_target_cols - 1:
                        point1 = corners2[c * self.cal_target_rows + r][0]
                        point2 = corners2[(c + 1) * self.cal_target_rows + r][0]
                        distance = np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
                        print('testing corners2[A][B]... || ' + 
                              ' / Point 1 | [' + str(r * self.cal_target_cols + c) + '][0] : ' + str(point1) + 
                              ' / Point 2 | [' + str(r * self.cal_target_cols + c + 1) + '][0]: ' + str(point2) + 
                              ' / Distance : ' + str(distance)
                              )
                        total_horizontal_distance += distance
                        num_horizontal_distances += 1
                        
                        x1 = int(point1[0])
                        y1 = int(point1[1])
                        x2 = int(point2[0])
                        y2 = int(point2[1])                       
                        
                        center1 = (x1, y1)
                        center2 = (x2, y2)
                        
                        cv.circle(self.gray_color_image, center1, 15, (255, 127, 127), 5)
                        cv.line(self.gray_color_image, center1, center2, (255, 127, 127), 5)
                        cv.circle(self.gray_color_image, center2, 15, (255, 127, 127), 5)
                        
                        cv.imshow(self.window_name, self.gray_color_image) 
                        
                        cv.waitKey(self.wait_time)
                    if r < self.cal_target_rows - 1:
                        point1 = corners2[c * self.cal_target_rows + r][0]
                        point2 = corners2[c * self.cal_target_rows + r + 1][0]
                        distance = np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
                        print('testing corners2[A][B] || ' + 
                              ' / Point 1 | [' + str(r * self.cal_target_cols + c) + '][0] : ' + str(point1) + 
                              ' / Point 2 | [' + str((r + 1) * self.cal_target_cols + c) + '][0]: ' + str(point2) + 
                              ' / Distance : ' + str(distance)
                              )
                        total_vertical_distance += distance
                        num_vertical_distances += 1 
                        
                        x1 = int(point1[0])
                        y1 = int(point1[1])
                        x2 = int(point2[0])
                        y2 = int(point2[1])                       
                        
                        center1 = (x1, y1)
                        center2 = (x2, y2)
                        
                        cv.circle(self.gray_color_image, center1, 7, (127, 127, 255), 5)
                        cv.line(self.gray_color_image, center1, center2, (127, 127, 255), 5)
                        cv.circle(self.gray_color_image, center2, 7, (127, 127, 127), 5)
                        
                        cv.imshow(self.window_name, self.gray_color_image) 
                        
                        cv.waitKey(self.wait_time)
                        
            average_horizontal_distance = total_horizontal_distance / num_horizontal_distances
            average_vertical_distance = total_vertical_distance / num_vertical_distances 
            average_distance = ( average_horizontal_distance + average_vertical_distance ) / 2
            self.mm_pixel_ratio = self.grid_square_mm / average_distance
            
            print('mm to pixel ratio : ' + str(self.mm_pixel_ratio))
        
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, self.gray.shape[::-1], None, None)
        
        self.gray_color_image = cv.cvtColor(self.gray, cv.COLOR_GRAY2BGR)
        h, w = self.gray_color_image.shape[:2]
                
        undistorted = cv.undistort(self.gray_color_image, mtx, dist, None)
        
        cv.waitKey(0)
        
        cv.imshow(self.window_name, undistorted)
        
        print('mtx - Camera Matrix: ')
        print(mtx)
        print('dist - Distortion Coefficients: ')
        print(dist)
        print('rvecs - Rotation Vectors:')
        print(rvecs)
        print('tvecs - Translation Vectors:')
        print(tvecs)
        
        # wait for input
        while (True):         
            
            input_key = cv.waitKey(self.wait_time)
            
            if input_key & 0xFF == 27:
                break
            elif cv.getWindowProperty(self.window_name, cv.WND_PROP_VISIBLE) < 1:
                break 
            
        cv.destroyAllWindows()
        
        return 0 
        
        
if __name__ == "__main__":
    tool = CameraCalibration()
    tool.calibrate()