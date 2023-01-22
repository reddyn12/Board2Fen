from operator import mod
import modelStruct

model = modelStruct.SimpleNet()
#modelStruct.loadModel(model, "models/simpModel2.pth")

total_params = sum(
	param.numel() for param in model.parameters()
)

trainable_params = sum(
	p.numel() for p in model.parameters() if p.requires_grad
)
cnt = 0

for i in model.parameters():
    cnt+=1


print(cnt, "model params")

print(total_params, "total params")

print(trainable_params, "trainable params")
