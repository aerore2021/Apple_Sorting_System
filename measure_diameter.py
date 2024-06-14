import datetime
import cv2
import os
import numpy as np
"""
In this file, we will measure the diameter of the apple using a single frame of the video stream.
So a whole apple shall be included in the image.
"""
class Diameter_Analyzer:
    def __init__(self, img: np.ndarray) -> None:
        # This methods may not be the best way to process the image, since it'll save the image of every step, which may be a waste of memory.
        # But it's a good way to debug every process, so I keep it here.
        self.img = img
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) # convert to gray scale
        self.blur = cv2.medianBlur(self.gray, 25) # Median blur to reduce noise and smooth the image
        self.mask = np.ndarray
        self.canny = np.ndarray
        self.diameters = 0.0
        self.bug = 0
        self.if_bug = False
    
    def segment(self):
        # HSV color space is unstable, which is interfered greatly by light, so we use gray scale to segment the image.
        # 直播流设置
        img = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8)
         # 取决于摄像头的位置，我们需要把传送带的部分截掉
        img[0:40, :] = 255
        img[-70:, :] = 255
        # 图片/视频设置
        # _, img = cv2.threshold(self.gray, 100, 255, cv2.THRESH_BINARY)
        self.mask = img
        # Opening operation to remove noise.
        # We tried closing operation, but it's not much benefitting.
        iter = 5
        kernal = np.ones((1, 1), np.uint8)
        img_open = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernal, iterations=20)
        img_open = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernal, iterations=15)
        img_canny = cv2.Canny(img_open, 70, 140)
        self.canny = img_canny
        # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_open)
        # min_size = 400
        # filtered_img = np.zeros_like(img_canny)

        # for i in range(1, num_labels):
        #     t = stats[i, cv2.CC_STAT_AREA]
        #     if stats[i, cv2.CC_STAT_AREA] >= min_size:
        #         filtered_img[labels == i] = 255
       
        
        self.mask = img_open
    
    def recs(self, Coordinates, states=0):    
        # Draw all rectangles on the image.
        for (x, y, diameter) in Coordinates:
            self.rec(x, y, diameter/2, states)
    
    def rec(self, x, y, radius, states):
        # Draw a rectangle around the apple, and its worm-eaten part.
        # We have more precise contours for computing the diameter, but leave the screen with a rectangle to show the result. That's enough.
        start_point = (int(x - radius), int(y - radius))
        end_point = (int(x + radius), int(y + radius))
        color = (0, 0, 255)
        thickness = 2
        # rectanlging will cover the original image.
        if states == 0:
            self.img = cv2.rectangle(self.img, start_point, end_point, color, thickness)
        elif states == 1:
            self.gray = cv2.rectangle(self.gray, start_point, end_point, color, thickness)
        elif states == 2:
            self.mask = cv2.rectangle(self.mask, start_point, end_point, color, thickness)
        elif states == 3:
            self.canny = cv2.rectangle(self.canny, start_point, end_point, color, thickness)
            
    def find_contours(self):
        self.segment()
        # Our processed image is self.canny, so we find contours on it.
        contours, _ = cv2.findContours(self.mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if_apple = False
        Apple_Coordinates = [] # store the coordinates of the apple.
        Coordinates = [] # store the coordinates of all contours.
        bugs = 0
        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            # print(x,y)
            # stream live version
            # 这里应该还要加一个苹果的y坐标，进一步限制传送带的影响
            if cv2.contourArea(contour) > 1000:
                if_apple = True
                Apple_Coordinates.append((x, y, radius*2))
                Coordinates.append((x, y, radius*2))
            elif 500 > cv2.contourArea(contour) > 100:
                bugs += 1
                Coordinates.append((x, y, radius*2))
            
        if self.if_bug == False:
            self.bug = bugs 
        return if_apple, Coordinates, Apple_Coordinates

# Testing code
if __name__ == '__main__':
    apples_path = './dataset/test/rottenapples'
    apples = [os.path.join(apples_path, apple_path) for apple_path in os.listdir(apples_path)]
    img_apples = [cv2.imread(apple) for apple in apples]
    
    for apple in img_apples:
        analyzer = Diameter_Analyzer(img=apple)
        analyzer.find_contours()
        # show image
        cv2.imshow('apple', analyzer.open)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()