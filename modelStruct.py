import torch
'''import torchvision
import numpy as np
import matplotlib.pyplot as plt'''
import torch.nn as nn
'''import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from PIL import Image'''
import torch.nn.functional as F


class ImgClassBase(nn.Module):
    def training_step(self, batch):
        #print(torch.cuda.memory_summary())
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        nn.CrossEntropyLoss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        #print("batchLosses", batch_losses, type(batch_losses))
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
        
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# tensor - (3, 50, 50)
class PieceNet(ImgClassBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 64, 26,26 for training

            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            #128, 14, 14 for training

            nn.Conv2d(128, 256, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 256, 8, 8

            nn.Flatten(),
            # This number matters as it is affected by the trainins img size.... I THINK

            nn.Linear(256*8*8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 13)
            
        )

    def forward(self, x):
        return self.network(x)

class SimpleNet(ImgClassBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 64, 26,26 for training

            

            nn.Flatten(),
            # This number matters as it is affected by the trainins img size.... I THINK

            nn.Linear(64*26*26, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 13)
            
        )

    def forward(self, x):
        return self.network(x)

def getDefaultDevice():
    
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
        
    else:
        return torch.device("cpu")

def toDevice(data, device):
    if isinstance(data, (list, tuple)):
        return[toDevice(x, device) for x in data]
    return data.to(device, non_blocking=True)
    
class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __iter__(self):
        for x in self.dl:
            yield toDevice(x, self.device)
    def __len__(self):
        return len(self.dl)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    #outputs = [model.validation_step(batch) for batch in val_loader]
    outputs = []
    cnt = 0
    for batch in val_loader:
        if cnt==50:
            break
        cnt = cnt + 1
        
        #print("evaluation batch cntr:", cnt)
        outputs.append(model.validation_step(batch))
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        print("running epoch: ", epoch)
        # Training Phase 
        model.train()
        train_losses = []
        cnt = 0
        le = len(train_loader)
        for batch in train_loader:
            #print("epoch ", epoch, "batch ", cnt, "/", le)
            loss = model.training_step(batch)
            
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            cnt= cnt + 1
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def saveModel(model, name):
    torch.save(model.state_dict(), "models/"+name+".pth")

def loadModel(model, path):
    model.load_state_dict(torch.load(path))

def predImg(model, img, device, classes):
    x = toDevice(img.unsqueeze(0), device)
    y = model(x)

    _, pred = torch.max(y, dim=1)
    #Dataset.classes
    return classes[pred[0].item()]






