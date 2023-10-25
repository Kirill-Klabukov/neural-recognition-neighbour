#about torch...
import torch
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset

#using numpy
import numpy as np

#for data load or save
import pandas as pd

#visualize some datasets
import matplotlib.pyplot as plt

#check our work directory
import os



lr = 0.001 # learning_rate
batch_size = 100 # we will use mini-batch method
epochs = 200 # How much to train a model
train = False

# os.listdir('data/train')

device = 'cuda' 

torch.manual_seed(1234)
if device == 'cuda':
    torch.manual_seed(1234)
    # torch.mps.manual_seed_all(1234)
    # os.makedirs('data/catORdog', exist_ok=True)
    base_dir = 'archive'
    train_dir = 'archive/seg_train'
    test_dir = 'archive/seg_pred'
    train_list = []
    test_list = []
    from pathlib import Path
    for path in Path(train_dir).rglob('*.png'):
        train_list.append(path)
    for path in Path(test_dir).rglob('*.png'):
        test_list.append(path)

    from PIL import Image

    random_idx = np.random.randint(1, 12000, size=10)


    from sklearn.model_selection import train_test_split
    train_list, val_list = train_test_split(train_list, test_size=0.2)

    # data Augumentation
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])


    class dataset(torch.utils.data.Dataset):

        def __init__(self, file_list, transform=None):
            self.file_list = file_list
            self.transform = transform

        # dataset length
        def __len__(self):
            self.filelength = len(self.file_list)
            return self.filelength

        # load an one of images
        def __getitem__(self, idx):
            img_path = self.file_list[idx]
            img = Image.open(img_path)
            img_transformed = self.transform(img)
            
            label = img_path.parts
            if img_path.parts[1] == 'seg_train':
                if label[2] == 'protogen':
                    label = 0
                elif label[2] == 'rexonium':
                    label = 1
                elif label[2] == 'beast':
                    label = 2
                elif label[2] == 'taidum':
                    label = 3
               
            else:
                
                label = label[2].split('.')[0]+'.'+label[2].split('.')[1]
            return img_transformed, label


    train_data = dataset(train_list, transform=train_transforms)
    test_data = dataset(test_list, transform=test_transforms)
    val_data = dataset(val_list, transform=test_transforms)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)


    class Cnn(nn.Module):
        def __init__(self):
            super(Cnn, self).__init__()

            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

            self.layer3 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

            self.fc1 = nn.Linear(3 * 3 * 64, 254)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(254, 120)
            self.fc3 = nn.Linear(120, 80)
            self.fc4 = nn.Linear(80, 40)
            self.fc5 = nn.Linear(40, 4)
            
            

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = self.fc2(out)
            out = self.fc3(out)
            out = self.fc4(out)
            out = self.fc5(out)
            return out


    model = Cnn().to(device)
    if os.path.isfile("CNNmodel.pt"):
        model.load_state_dict(torch.load('CNNmodel.pt'))
    model.train()

    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    
    lastAccuracy = 0
    nowAccuracy = 0
    if train == True :
        print("Train Starts")
        for epoch in range(epochs):
            print("Epoch Starts")
            epoch_loss = 0
            epoch_accuracy = 0
        

            # if os.path.isfile("CNNmodel.pt"):
            #     model.load_state_dict(torch.load('CNNmodel.pt'))
            for data, label in train_loader:
                data = data.to(device)
                
                label = label.to(device)
                

                output = model(data)
                
                loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = ((output.argmax(dim=1) == label).float().mean())
                epoch_accuracy += acc / len(train_loader)
                epoch_loss += loss / len(train_loader)

            print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch + 1, epoch_accuracy, epoch_loss))

            with torch.no_grad():
                epoch_val_accuracy = 0
                epoch_val_loss = 0
                for data, label in val_loader:
                    data = data.to(device)
                
                    label = label.to(device)

                    val_output = model(data)
                    val_loss = criterion(val_output, label)

                    acc = ((val_output.argmax(dim=1) == label).float().mean())
                    epoch_val_accuracy += acc / len(val_loader)
                    epoch_val_loss += val_loss / len(val_loader)
                nowAccuracy = 100 * epoch_val_accuracy   
                print('Epoch : {}, val_accuracy : {}, val_loss : {}'.format(epoch + 1, epoch_val_accuracy, epoch_val_loss))
                print(f'Accuracy of the network on the 10000 test images: {100 * epoch_val_accuracy} %')
            if(nowAccuracy > lastAccuracy):
                torch.save(model.state_dict(), 'CNNmodel.pt')
                print("Progress Saved")
                lastAccuracy = nowAccuracy
    

    if train == False:
        class_ = {0: 'protogen', 1: 'rexonium', 2: 'beast', 3: 'taidum'}
        results = []
        model.eval()
        with torch.no_grad():
            for data, fileid in test_loader:
                data = data.to(device)
                preds = model(data)
                
                _, preds = torch.max(preds, dim=1)
                
                results += list(zip(list(fileid), preds.tolist()))
                

       
       
       
        idx = list(map(lambda x: x[0], results))
        prob = list(map(lambda x: x[1], results))

        submission = pd.DataFrame({'id': idx, 'label': prob})
        
        

        import random

        id_list = []
        
        
        fig, axes = plt.subplots(2, 5, figsize=(5, 5), facecolor='w')
       

        for ax in axes.ravel():

            i = random.choice(submission['id'].values)
           
            label = submission.loc[submission['id'] == i, 'label'].values[0]
            
            

            img_path = os.path.join(test_dir, '{}.png'.format(i))
            img = Image.open(img_path)
            img = img.resize((224, 224))
            ax.set_title(class_[label])
            ax.axis('off')
            ax.imshow(img)
           
        
        plt.show()
    
       
        
