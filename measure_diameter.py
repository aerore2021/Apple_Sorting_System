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
        self.blur = cv2.GaussianBlur(self.img, (7, 7), 0) # Gaussian blur to reduce noise and smooth the image
        self.gray = cv2.cvtColor(self.blur, cv2.COLOR_BGR2GRAY) # convert to gray scale
        self.mask = None
        self.canny = None
        self.diameter = None
    
    def segment(self):
        # HSV color space is unstable, which is interfered greatly by light, so we use gray scale to segment the image.
        # 直播流设置
        # _, img = cv2.threshold(self.gray, 170, 255, cv2.THRESH_BINARY)
        # 图片/视频设置
        _, img = cv2.threshold(self.gray, 100, 255, cv2.THRESH_BINARY)
        self.mask = img
        # Opening operation to remove noise.
        # We tried closing operation, but it's not much benefitting.
        iter = 5
        kernal = np.ones((3, 3), np.uint8)
        img_open = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernal, iterations=iter)
        img_canny = cv2.Canny(img_open, 100, 200)
        self.canny = img_canny
    
    def rec(self, x, y, radius):
        # Draw a rectangle around the apple, and its worm-eaten part.
        # We have more precise contours for computing the diameter, but leave the screen with a rectangle to show the result. That's enough.
        start_point = (int(x - radius), int(y - radius))
        end_point = (int(x + radius), int(y + radius))
        color = (0, 0, 255)
        thickness = 2
        # rectanlging will cover the original image.
        self.img = cv2.rectangle(self.img, start_point, end_point, color, thickness)
   
    def find_contours(self):
        self.segment()
        # Our processed image is self.canny, so we find contours on it.
        contours, _ = cv2.findContours(self.canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if_apple = False
        Apple_Coordinates = [] # store the coordinates of the apple.
        Coordinates = [] # store the coordinates of all contours.
        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            Coordinates.append((x, y, radius*2))
            # stream live version
            # 视频版本的坐标是(270, 370)，仅用作debug
            # 这里应该还要加一个苹果的y坐标，进一步限制传送带的影响
            if cv2.contourArea(contour) > 10000 and 1250 <= x <= 1350:
                if_apple = True
                Apple_Coordinates.append((x, y, radius*2))
                print("Apple found, recongnizing...")
            
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