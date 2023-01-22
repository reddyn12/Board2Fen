#%%
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

#%%
x = torch.rand(5, 3)
print(x)
s = "stinged"

torch.AggregationType.AVG
# %%
test = [[11,22,33,44],[55,66,77,88,99],[12,23,34,45,56]]

print(test[1][0])
