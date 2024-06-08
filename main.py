import cv2 as cv
import torch.utils
import measure_diameter as md
import train_model as tm
import torch
import torch.nn as nn
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import warnings

if __name__ == '__main__':
    # ignore wanrnings raised by torchvision. 
    warnings.filterwarnings("ignore")
    # load local model
    if os.path.exists('apple_model.pth'):
        print("Model found, loading...")
        model_dict = torch.load('apple_model.pth')
        apple_trainer = tm.Train_ResNet()
        apple_trainer.model_construction()
        apple_trainer.load_dataset('dataset')
        apple_model = apple_trainer.model
        apple_model.load_state_dict(model_dict)
        print("Model loaded successfully.")
        # if the model doesn't exit, train one from the start. 
    else:
        print("Model not found, training a new model...")
        apple_trainer = tm.Train_ResNet()
        apple_trainer.load_dataset('dataset')
        apple_trainer.model_construction()
        apple_trainer.training(epochs=10)
        apple_model = apple_trainer.model
        torch.save(apple_trainer.model.state_dict(), 'apple_model.pth')
        print("Model trained and saved successfully.")

    
    # camera capture
    print("Opening camera...")
    cap = cv.VideoCapture(0)
    # cap = cv.VideoCapture("apple_5_31.mp4")
    cv.namedWindow('apples', cv.WINDOW_NORMAL)
    # frame reshape setting for ResNet
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    ripeness_weight = torch.Tensor([0.3, 0.7])
    ripeness = 0.0
    if_rotten = []
    diameters = []
    last_if_apple = False
    while cap.isOpened():
        ret, frame = cap.read()
        # show image
        if ret:
            analyzer = md.Diameter_Analyzer(img=frame)
            if_apple, coordinates, apple_coordinates = analyzer.find_contours()
            for (x, y, diameter) in coordinates:
                analyzer.rec(x,y,diameter/2)
            cv.imshow('apples', analyzer.img)
            
            if if_apple:
                # Capture the apple and send it to the model for prediction.
                (x, y, diameter) = apple_coordinates[0]
                frame_apple = frame[int(y-diameter/2):int(y+diameter/2), int(x-diameter/2):int(x+diameter/2)]
                frame_apple_int8 = Image.fromarray(np.uint8(frame_apple))
                frame_CNN = transform(frame_apple_int8)
                
                outputs = apple_model(frame_CNN.to(apple_trainer.device).unsqueeze(0))
                outputs = nn.Softmax(dim=1)(outputs)
                ripeness = torch.mul(outputs, ripeness_weight)
                ripeness = torch.sum(ripeness, dim=1)
                diameters.append(diameter)
                # print(torch.max(outputs.data, 1).indices.item(), diameter)
            # 这是stream的坐标版本
            if last_if_apple == True and if_apple == False:
                print("Apple ripeness: ", ripeness.item(), "Diameter: ", diameter)
                if_rotten = []
                diameters = []
            last_if_apple = if_apple
                
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()