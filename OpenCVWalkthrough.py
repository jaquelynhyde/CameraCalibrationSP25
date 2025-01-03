# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 17:33:22 2024

@author: Jaque
"""

import numpy as np
import cv2 as cv

class OpenCVWalkthrough:
    def __init__(self):
        self.input_image = cv.imread(cv.samples.findFile("Input_Image.jpg"))
        self.gray_image = None
        self.gray_color_image = None
        self.edges_image = None
        self.edges_color_image = None
        self.contours = None
        self.hierarchy = None
        self.mm_pixel_ratio = 1
        self.window_name = "Open CV Walkthrough"
        
    # Given:    An image
    # Return:   An edge-detected image    
    @staticmethod
    def detect_edges(image, min_val, max_val):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
        edges = cv.Canny(gray, min_val, max_val)
        
        return edges
       
    def walkthrough(self):
        wait_time = 25 
        
        min_val = 127
        max_val = 200 
        
        cv.namedWindow(self.window_name)
        
        image_height, image_width, channels = self.input_image.shape
        
        cv.resizeWindow(self.window_name, image_height, image_width)
     
        self.gray_image = cv.cvtColor(self.input_image, cv.COLOR_BGR2GRAY)
        
        self.gray_color_image = cv.cvtColor(self.gray_image, cv.COLOR_GRAY2BGR)
        
        cv.imshow(self.window_name, self.gray_color_image)  
        
        cv.waitKey(0)
        
        rows = 7
        cols = 10
        grid_square_mm = 15

        termination_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        objp = np.zeros((rows*cols,3), np.float32)
        objp[:,:2] = np.mgrid[0:rows,0:cols].T.reshape(-1,2)
        
        objpoints = [] # points in 3d space (real world)
        imgpoints = [] # points in 2d space (image plane)
                
        ret, corners = cv.findChessboardCorners(self.gray_image, (7, 10), None)
        
        if ret == True:
            objpoints.append(objp)
            
            corners2 = cv.cornerSubPix(self.gray_image, corners, (11,11), (-1,-1), termination_criteria)
            
            print('Corners Array :')
            print(corners2)
            
            imgpoints.append(corners2)
            
            total_horizontal_distance = 0
            total_vertical_distance = 0
            num_horizontal_distances = 0
            num_vertical_distances = 0
            
            for r in range(rows):
                print('testing row ' + str(r))
                for c in range(cols):
                    print('testing column ' + str(c))
                    
                    if c < cols - 1:
                        point1 = corners2[c * rows + r][0]
                        point2 = corners2[(c + 1) * rows + r][0]
                        distance = np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
                        print('testing corners2[A][B]... || ' + 
                              ' / Point 1 | [' + str(r * cols + c) + '][0] : ' + str(point1) + 
                              ' / Point 2 | [' + str(r * cols + c + 1) + '][0]: ' + str(point2) + 
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
                        
                        cv.circle(self.gray_color_image, center1, 15, (255, 0, 0), 3)
                        cv.line(self.gray_color_image, center1, center2, (255, 0, 0), 3)
                        cv.circle(self.gray_color_image, center2, 15, (255, 0, 0), 3)
                        
                        cv.imshow(self.window_name, self.gray_color_image) 
                        
                        cv.waitKey(wait_time)
                    if r < rows - 1:
                        point1 = corners2[c * rows + r][0]
                        point2 = corners2[c * rows + r + 1][0]
                        distance = np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
                        print('testing corners2[A][B] || ' + 
                              ' / Point 1 | [' + str(r * cols + c) + '][0] : ' + str(point1) + 
                              ' / Point 2 | [' + str((r + 1) * cols + c) + '][0]: ' + str(point2) + 
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
                        
                        cv.circle(self.gray_color_image, center1, 7, (0, 0, 255), 3)
                        cv.line(self.gray_color_image, center1, center2, (0, 0, 255), 3)
                        cv.circle(self.gray_color_image, center2, 7, (0, 0, 255), 3)
                        
                        cv.imshow(self.window_name, self.gray_color_image) 
                        
                        cv.waitKey(wait_time)
                        
            average_horizontal_distance = total_horizontal_distance / num_horizontal_distances
            average_vertical_distance = total_vertical_distance / num_vertical_distances 
            average_distance = ( average_horizontal_distance + average_vertical_distance ) / 2
            average_distance_mm = grid_square_mm / average_distance
            
            print('mm to pixel ratio : ' + str(average_distance_mm))
            
            self.mm_pixel_ratio = average_distance_mm
            
        
        
        self.edges_image = self.detect_edges(self.input_image, min_val, max_val)

        self.contours, self.hierarchy = cv.findContours(self.edges_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE) 
        
        print('----------------')
        print('Hierarchy : [next, previous, first child, parent] ')
        print('A 3D array, A by B by C, A = ' + str(len(self.hierarchy))
              + ', B = ' + str(len(self.hierarchy[0]))
              + ', C = 4')
        print(self.hierarchy)
        print('----------------')

        self.edges_color_image = cv.cvtColor(self.edges_image, cv.COLOR_GRAY2BGR)
        
        cv.drawChessboardCorners(self.edges_color_image, (rows, cols), corners2, ret)
        
        cv.imshow(self.window_name, self.edges_color_image)        
        
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, self.gray_image.shape[::-1], None, None)
        
        print('cv.calibrateCamera output: ')
        print('ret : ' + str(ret))
        print('mtx (intrinsic camera property matrix):')
        print(str(mtx))
        print('dist : ' + str(dist))
        print('rvecs : ' + str(rvecs))
        print('tvecs : ' + str(tvecs))
        
        blu_val = 33
        grn_val = 66
        red_val = 132
        
        blu_shift = 22
        grn_shift = 22
        red_shift = 66
        
        print('Displaying contours...')
        for i, contour in enumerate(self.contours):           
            
            M = cv.moments(self.contours[i])
            
            print('For contour ' + str(i) + ', Moments = ')
            print(M)
            
            if (M['m00'] != 0):
                x0 = M['m10']/M['m00']
                y0 = M['m01']/M['m00']
                center = (int(x0),int(y0))
                cv.circle(self.edges_color_image, center, 5, (blu_val, grn_val, red_val), 3)
                                    
            cv.drawContours(self.edges_color_image, self.contours, i, (blu_val, grn_val, red_val), thickness = 3)
                
            cv.imshow(self.window_name, self.edges_color_image)  
            red_val = red_val + red_shift
            blu_val = blu_val + blu_shift
            grn_val = grn_val + grn_shift
            
            if (red_val < 0):
                red_val = 0
                red_shift = -red_shift
            elif (red_val > 255):
                red_val = 255
                red_shift = -red_shift
                
            if (blu_val < 0):
                blu_val = 0
                blu_shift = -blu_shift
            elif (blu_val > 255):
                blu_val = 255
                blu_shift = -blu_shift
                
            if (grn_val < 0):
                grn_val = 0
                grn_shift = -grn_shift
            elif (grn_val > 255):
                grn_val = 255
                grn_shift = -grn_shift
                
            cv.waitKey(wait_time)
        
            
        
        while (True):         
            
            input_key = cv.waitKey(wait_time)
            
            if input_key & 0xFF == 27:
                break
            elif cv.getWindowProperty(self.window_name, cv.WND_PROP_VISIBLE) < 1:
                break 
            
        cv.destroyAllWindows()
        
        return 0 
        
if __name__ == "__main__":
    tool = OpenCVWalkthrough()
    tool.walkthrough()