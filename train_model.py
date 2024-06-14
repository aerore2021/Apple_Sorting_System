import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader  
from tqdm import tqdm
import warnings
from torchvision import transforms
from PIL import Image
import numpy as np

def model_image_process(frame, x, y, diameter):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    frame_apple = frame[int(y-diameter/2):int(y+diameter/2), int(x-diameter/2):int(x+diameter/2)]
    frame_apple_int8 = Image.fromarray(np.uint8(frame_apple))
    frame_CNN = transform(frame_apple_int8)
    return frame_CNN

# Apple Resnet
class My_AR:
    def __init__(self) -> None:
        self.model, self.trainer = self.load_or_train_model()
        self.ripeness = 0.0
        
    def load_or_train_model(self):
        # ignore wanrnings raised by torchvision. 
        warnings.filterwarnings("ignore")
        # load local model
        if os.path.exists('apple_model.pth'):
            print("Model found, loading...")
            model_dict = torch.load('apple_model.pth')
            apple_trainer = Train_ResNet()
            apple_trainer.model_construction()
            apple_trainer.load_dataset('dataset')
            apple_model = apple_trainer.model
            apple_model.load_state_dict(model_dict)
            print("Model loaded successfully.")
            # if the model doesn't exit, train one from the start. 
        else:
            print("Model not found, training a new model...")
            apple_trainer = Train_ResNet()
            apple_trainer.load_dataset('dataset')
            apple_trainer.model_construction()
            apple_trainer.training(epochs=10)
            apple_model = apple_trainer.model
            torch.save(apple_trainer.model.state_dict(), 'apple_model.pth')
            print("Model trained and saved successfully.")
        return apple_model, apple_trainer
    
    def predict(self, img):
        ripeness_weight = torch.Tensor([0.3, 0.7])
        outputs = self.model(img.to(self.trainer.device).unsqueeze(0))
        outputs = nn.Softmax(dim=1)(outputs)
        ripeness = torch.mul(outputs, ripeness_weight)
        ripeness = torch.sum(ripeness, dim=1)
        self.ripeness = ripeness.data.numpy()[0]
    
    
class Train_ResNet:
    def __init__(self) -> None:
        self.model = models.resnet50(pretrained=True)
        self.train_data = None
        self.test_data = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.criterion = None
    
    def load_dataset(self, data_dir):
        # ImageFolder会自动读取子文件夹的图片，并将文件夹名作为label
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        dataset = datasets.ImageFolder(os.path.join(data_dir,'train'), transform=transform)
        train_size = int(0.7 * len(dataset))
        test_size = len(dataset) - train_size
        tran_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        self.train_data = DataLoader(tran_dataset, shuffle=True, num_workers=2)
        self.test_data = DataLoader(test_dataset, shuffle=True, num_workers=2)
    
    def model_construction(self):
        # 苹果熟度的类别
        num_classes = 2
        num_features = self.model.fc.in_features
        # replacing the existing fully-connected layer with a 'num_features' input and 'num_classes' output features FC layer
        self.model.fc = nn.Linear(num_features, num_classes)
        # loss function
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
    
    def training(self, epochs=10):
        self.model.to(self.device)
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            running_corrects = 0
            print("Epoch %d:" % epoch)
            for data in tqdm(self.train_data, leave=True):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                # get loss value and update weights
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss/len(self.train_data)
            epoch_acc = running_corrects / len(self.train_data) * 100 /self.train_data.batch_size
            print('[Train #{}] Loss: {:.4f} Acc: {:.4f}%'.format(epoch, epoch_loss, epoch_acc))
            
    def testing(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_data,0):
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
            
# train
if __name__ == '__main__':
    if os.path.exists('apple_model.pth'):
        model_dict = torch.load('apple_model.pth')
        apple_trainer = Train_ResNet()
        apple_trainer.model_construction()
        apple_trainer.load_dataset('dataset')
        apple_model = apple_trainer.model
        apple_model.load_state_dict(model_dict)
        print(apple_model)
        apple_trainer.testing()
    else:
        trainer = Train_ResNet()
        trainer.load_dataset('dataset')
        trainer.model_construction()
        trainer.training(epochs=10)
        # print(trainer.model)
        trainer.testing()
        torch.save(trainer.model.state_dict(), 'apple_model.pth')
        
    # trainer = Train_ResNet()
    # print(trainer.model)