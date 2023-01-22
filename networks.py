#%%
from codecs import getdecoder
from pyexpat import model
from tempfile import TemporaryDirectory
from time import process_time_ns
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
#%%
class ImgClassBase(nn.Module):
    def training_step(self, batch):
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
#%%

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
#%%
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
    

'''

- make a new func to split the test data into a new folder
- see what the end func is for multi class - I think its softmax, or make array of classifiers thats true false

'''
# %%
n_epochs = 10
batch_size_train = 128
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

#%% 

trainData = torchvision.datasets.ImageFolder(root = "./dataSep", transform = ToTensor())
trainLoader = DataLoader(trainData, batch_size=batch_size_train, shuffle=True)
#%%
testData = torchvision.datasets.ImageFolder(root = "./dataSepTest", transform = ToTensor())
testLoader = DataLoader(testData, batch_size=batch_size_test, shuffle=True)

#%%

testimg, tlabel = trainData.__getitem__(100)
untransform = torchvision.transforms.ToPILImage()
testimg = untransform(testimg)
#testimg.show()



# %%
print(trainData[0][0].shape)
print(trainData.classes)

#%%
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(trainLoader.dataset) for i in range(n_epochs + 1)]
# %%
'''def train(epoch):
  model1.train()
  for batch_idx, (data, target) in enumerate(trainLoader):
    optimizer.zero_grad()
    output = model1(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(trainLoader.dataset),
        100. * batch_idx / len(trainLoader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(trainLoader.dataset)))
      torch.save(model1.state_dict(), '/results/model1.pth')
      torch.save(optimizer.state_dict(), '/results/optimizer1.pth')'''


# %%
'''def test():
  model1.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in testLoader:
      output = model1(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(testLoader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(testLoader.dataset),
    100. * correct / len(testLoader.dataset)))'''
# %%
'''test()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()'''

#%%
'''for images, labels in trainLoader:
    print("images.shape", images.shape)
    out = model1(images)
    
    print("out.shape", out.shape)
    print("out[0]:", out[0])
    outclean = F.softmax(out, dim=1)
    print("outclean", outclean)
 
    break'''
# %%
torch.cuda.is_available()

# %%
def getDefaultDevice():
    if torch.cuda.is_available():
        return torch.device("cuda")
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


# %%
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
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history
'''def train(model, trainLoader, testLoader, epochs, lr, optFunc=torch.optim.SGD):
    pass

@torch.no_grad
def test(model, testLoader):
    model.eval()'''
#%%

def saveModel(model, name):
    torch.save(model.state_dict(), "models/"+name+".pth")

# %%
device = getDefaultDevice()
print(device)
trainLoader = DeviceDataLoader(trainLoader, device)
testLoader = DeviceDataLoader(testLoader, device)
#toDevice(model1, device)
#%%
model1 = toDevice(PieceNet(), device)

# %%

evaluate(model1, testLoader)

# %%
print(len(testLoader))

# %%
simpModel = toDevice(SimpleNet(), device)

#%%

simpModel.load_state_dict(torch.load("models/simpleModel1.pth"))
# %%
evaluate(simpModel, testLoader)
# %%
historySimp = fit(10, learning_rate, simpModel, trainLoader, testLoader)

#%%
print("hi")
# %%
saveModel(simpModel, "simpleModel1")
# %%
def loadModel(model, path):
    model.load_state_dict(torch.load(path))

def predImg(model, img):
    x = toDevice(img.unsqueeze(0), device)
    y = model(x)

    _, pred = torch.max(y, dim=1)

    return trainData.classes[pred[0].item()]

#%%
print(trainData.classes)
for images, labels in trainLoader:
    print("images.shape", images.shape)
    img = images[0]
    print(labels[0])
    print(predImg(simpModel, img))
    #i = untransform(img)
    #i.show()
    
    
 
    break
# %%

tempImg = Image.open("custom/wHorse.jpeg")
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((50, 50)), ToTensor()])

tensImg = transform(tempImg)
tempImg.show()

print(predImg(simpModel,tensImg))

# %%
t = Image.open("dataSep/k/00b04c6b-f8ac-4074-b16b-a152ddac0399.jpeg")
print((t.size))
# %%
print(tensImg.shape)
# %%
tt = transform(t)
print(tt.shape)
print(predImg(simpModel,tt))
# %%
