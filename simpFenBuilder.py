#%%
import modelStruct
import os
from PIL import Image
import torchvision

#%%
simpModel = modelStruct.loadModel(modelStruct.SimpleNet(), "models/simpleModel1.pth")

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((50, 50)), torchvision.transforms.ToTensor()])

#%%

data = os.listdir("data/train")
#print(data[0])

#%%
def imgCheck(path, model):
    image = Image.open(path)
    
    #image.show()

    width, height = image.size
    #print(width, height, "w/h")

    # Calculate the number of rows and columns
    # num of pixels
    num_cols = width // 8
    num_rows = height // 8

    sub_images = []

    # Iterate over the rows and columns
    for row in range(8):
        for col in range(8):
            # Calculate the coordinates of the sub-image
            left = col * num_cols
            top = row * num_rows
            right = left + num_cols
            bottom = top + num_rows
            
            # Extract the sub-image
            sub_image = image.crop((left, top, right, bottom))

            imgTens = transform(sub_image)
            

            
            # Add the sub-image to the list
            #sub_images.append(sub_image)
    #print(len(sub_images))
    #sub_images[1].show()
    #print(f[0][1])

    
# %%
#do the check code
cnt = 0
r = 0
for fenB in data:
    fen=fenB.split(".")[0]
    print(fen)
    imgCheck(fenB, simpModel)
    break


# %%
