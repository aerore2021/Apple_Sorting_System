import sys
import numpy as np
import cv2
from measure_diameter import Diameter_Analyzer
from train_model import My_AR, model_image_process
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class Video():
    def __init__(self, capture):
        self.capture = capture
        self.currentFrame = np.ndarray([])
        self.analyzer = None
        self.Framestate = 0
    
    def updateFrameState(self, state):
        self.Framestate = state
    
    def captureFrame(self, analyzer:Diameter_Analyzer):
        if self.Framestate == 0:
            frame = analyzer.img
        elif self.Framestate == 1:
            frame = analyzer.gray
        elif self.Framestate == 2:
            frame = analyzer.mask
        elif self.Framestate == 3:
            frame = analyzer.canny
        return frame
    
    def captureNextFrame(self, ret, analyzer:Diameter_Analyzer):
        if ret:
            if self.Framestate == 0:
                self.currentFrame = cv2.cvtColor(analyzer.img, cv2.COLOR_BGR2RGB)
            elif self.Framestate == 1:
                self.currentFrame = analyzer.gray # since we already done the convert process.
            elif self.Framestate == 2:
                self.currentFrame = analyzer.mask
            elif self.Framestate == 3:
                self.currentFrame = analyzer.canny
    
    def convertFrame(self):
        try:
            height, width = self.currentFrame.shape[:2]
            if self.Framestate == 0:
                img = QImage(self.currentFrame, width, height, QImage.Format_RGB888)
            elif self.Framestate == 1:
                img = QImage(self.currentFrame, width, height, QImage.Format_Grayscale8)
            elif self.Framestate == 2:
                img = QImage(self.currentFrame, width, height, QImage.Format_Grayscale8)
            elif self.Framestate == 3:
                img = QImage(self.currentFrame, width, height, QImage.Format_Grayscale8)
            img = QPixmap.fromImage(img)
            self.previousFrame = self.currentFrame
            return img
        except:
            return None
            
class MyGui(QWidget):
    def __init__(self):
        super().__init__()
        self.Framestate = 0
        self.resize(800, 600)
        self.setWindowTitle('APPLE')
        self.setGeometry(100, 100, 800, 600)
        self.layout_init()
        self.AR = My_AR
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.display)
        self._timer.start(27)
        
        
    def layout_init(self):
       self.__layout_main = QVBoxLayout()
       self.video_init()
       self.fun_btn_init()
       self.results_init()
       self.__layout_main.addWidget(self.videoFrame)
       self.__layout_main.addLayout(self.__layout_fun_btn)
       self.__layout_main.addLayout(self.__layout_results)
       self.setLayout(self.__layout_main)
        
    
    def video_init(self):
        self.video = Video(cv2.VideoCapture(0))
        self.videoFrame = QLabel('capture')
        self.videoFrame.setAlignment(Qt.AlignCenter)
        # self.setCentralWidget(self.videoFrame)
        self.ret, self.frame = self.video.capture.read()
        self.video.analyzer = Diameter_Analyzer(self.frame)
        
    def fun_btn_init(self):
        self.__layout_fun_btn = QHBoxLayout()        
        # layout function buttons:
        ## stands for button of original image
        self.ImgBtnOr = QPushButton('Original', self)
        self.__layout_fun_btn.addWidget(self.ImgBtnOr)
        ## stands for button of gray image
        self.ImgBtnGr = QPushButton('Gray', self)
        self.__layout_fun_btn.addWidget(self.ImgBtnGr)
        ## stands for button of binary image
        self.ImgBtnBi = QPushButton('Binary', self)
        self.__layout_fun_btn.addWidget(self.ImgBtnBi)
        ## stands for button of edge image
        self.ImgBtnEd = QPushButton('Edge', self)
        self.__layout_fun_btn.addWidget(self.ImgBtnEd)
        # button pressed
        self.ImgBtnOr.clicked.connect(lambda: self.on_button_clicked(self.ImgBtnOr))
        self.ImgBtnGr.clicked.connect(lambda: self.on_button_clicked(self.ImgBtnGr))
        self.ImgBtnBi.clicked.connect(lambda: self.on_button_clicked(self.ImgBtnBi))
        self.ImgBtnEd.clicked.connect(lambda: self.on_button_clicked(self.ImgBtnEd))
    
    def results_init(self):
        self.__layout_results = QVBoxLayout()
        # layout results
        self.diameter_result = QLabel('Diameter: '+str(0.0))
        self.ripeness_result = QLabel('Ripeness: '+str(0.0)+'%')
        self.__layout_results.addWidget(self.diameter_result)
        self.__layout_results.addWidget(self.ripeness_result)
    
    def display(self):
        try:
            self.ret, self.frame = self.video.capture.read()
            self.video.analyzer = Diameter_Analyzer(self.frame)
            if_apple, coordinates, apple_coordinates = self.video.analyzer.find_contours()    
            self.results_update(if_apple, apple_coordinates)
            self.video.updateFrameState(self.Framestate)
            self.video.analyzer.recs(coordinates, states=self.Framestate)
            self.video.captureNextFrame(self.ret, self.video.analyzer)
            self.videoFrame.setPixmap(self.video.convertFrame())
        except:
            print("No frame")
    
    def results_update(self, if_apple, apple_coordinates):
        if if_apple == True:
            # Capture the apple and send it to the model for prediction.
            (x, y, diameter) = apple_coordinates[0]
            img_resnet = model_image_process(self.frame, x, y, diameter)
            self.AR.predict(img_resnet)
            self.video.analyzer.diameter=diameter
            self.diameter_result.setText('Diameter: '+str(self.video.analyzer.diameter))
            self.ripeness_result.setText('Ripeness: '+str(self.AR.ripeness * 100)+'%')
    
    def on_button_clicked(self, btn):
        if btn.text() == 'Original':
            self.Framestate = 0
        elif btn.text() == 'Gray':
            self.Framestate = 1
        elif btn.text() == 'Binary':
            self.Framestate = 2
        elif btn.text() == 'Edge':
            self.Framestate = 3


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = MyGui()
    gui.show()
    sys.exit(app.exec_())