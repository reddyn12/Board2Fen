#%%
from functools import total_ordering
import modelStruct
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision.transforms import ToTensor
import time
import torch
'''
If reserved memory is >> allocated memory try 
setting max_split_size_mb to avoid fragmentation.  
See documentation for Memory Management and 
PYTORCH_CUDA_ALLOC_CONF

'''
b = time.time()

print("start: ", b)

n_epochs = 10
batch_size_train = 128 *4  #128
batch_size_test = 1000 * 1
learning_rate = 0.01
momentum = 0.5
log_interval = 10

#%% 
print("loading train data")
trainData = torchvision.datasets.ImageFolder(root = "./dataSep", transform = ToTensor())
trainLoader = DataLoader(trainData, batch_size=batch_size_train, shuffle=True)
#%%
print("loading test data")
testData = torchvision.datasets.ImageFolder(root = "./dataSepTest", transform = ToTensor())
testLoader = DataLoader(testData, batch_size=batch_size_test, shuffle=True)

'''#%%
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(trainLoader.dataset) for i in range(n_epochs + 1)]'''


#%%
device = modelStruct.getDefaultDevice()
print("device is : ", device)
trainLoader = modelStruct.DeviceDataLoader(trainLoader, device)
testLoader = modelStruct.DeviceDataLoader(testLoader, device)
print("dta loaded to: ", device)

#%%

simpModel = modelStruct.toDevice(modelStruct.SimpleNet(), device)
print("model loaded to:", device)
for i in simpModel.parameters():
    print(i.is_cuda)
print(trainLoader.device)
print(torch.cuda.memory_allocated(), "mem allocated")
print(torch.cuda.memory_reserved(),"mem cached")
print(torch.cuda.memory_summary())
'''device_index = torch.cuda.current_device()
    # Get the name of the current GPU
device_name = torch.cuda.get_device_name(device_index)
# Print the name of the current GPU
print(f'Current GPU: {device_name}')
# Get the list of tasks running on the GPU
tasks = torch.cuda.list_gpu_processes(device)
# Print the list of tasks
print(f'Tasks on GPU: {tasks}')'''
'''# %%
modelStruct.evaluate(simpModel, testLoader)'''
# %%
#print("starting model on: ", simpModel.device)
historySimp = modelStruct.fit(2, learning_rate, simpModel, trainLoader, testLoader)

a = time.time()

print(a-b)

#modelStruct.saveModel(simpModel, "simpModel2")

