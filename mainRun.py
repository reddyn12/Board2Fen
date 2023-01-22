#%%

import tempfile
import torch
import torchvision
import numpy as np
from torch import conv3d
import os
import uuid
from PIL import Image
#%%

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
#%%

trainPathArr = os.listdir("data/test")
trainfen = []
for i in trainPathArr:
    trainfen.append(i.split(".")[0])
# %%

print(trainfen[0:10])
# %%
def dataPreProcess(path):
    pathArr = os.listdir(path)
    print("this shit hit123")
    pieces = "prbnqkPRBNQK"
    fen = []
    fenBig = []
    fenBigDash = []

    for i in pathArr:
        temp = i.split(".")[0]
        fen.append(temp)
        tempBig = ""
        tempBigDash = ""
        for j in temp.split("-"):
            for k in j:
                if(k in pieces):
                    tempBig += k
                    tempBigDash += k
                else:
                    for x in range(int(k)):
                        tempBig += "x"
                        tempBigDash += "x"
            tempBigDash += "-"
        fenBig.append(tempBig)
        fenBigDash.append(tempBigDash)

    
        

    
    
    
    return pathArr, fen, fenBig, fenBigDash

#%%

p, f, fb, fbd = dataPreProcess("data/test")
print(f[0:10])
print(fb[0:10])
print(fbd[0:10])
print(len(fb[0]))

# %%


# Open the image
image = Image.open("data/test/1b1B1Qr1-7p-6r1-2P5-4Rk2-1K6-4B3-8.jpeg")
image.show()

width, height = image.size
print(width, height, "w/h")

# Calculate the number of rows and columns
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
        
        # Add the sub-image to the list
        sub_images.append(sub_image)
print(len(sub_images))
sub_images[1].show()
print(f[0][1])
for i, sub_image in enumerate(sub_images):
    # Save the sub-image to disk
    pass
    #sub_image.save("sub_image_{}.jpg".format(i))

# %%

def imgParse(path):
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
            
            # Add the sub-image to the list
            sub_images.append(sub_image)
    #print(len(sub_images))
    #sub_images[1].show()
    #print(f[0][1])

    return sub_images
    

# %%



def makeNewDataFormat(path):
    
    p, f, fb, fbd = dataPreProcess(path)

    for i, fen in enumerate(fb):
        print(i, fen)
        imgs = imgParse(path + "/" + p[i])
        for j,img in enumerate(imgs):
            if(fen[j]=="p"):
                img.save("dataSepTest/p/" + str(uuid.uuid4()) + ".jpeg")
            elif(fen[j]=="b"):
                img.save("dataSepTest/b/" + str(uuid.uuid4()) + ".jpeg")
            elif(fen[j]=="n"):
                img.save("dataSepTest/n/" + str(uuid.uuid4()) + ".jpeg")
            elif(fen[j]=="r"):
                img.save("dataSepTest/r/" + str(uuid.uuid4()) + ".jpeg")
            elif(fen[j]=="q"):
                img.save("dataSepTest/q/" + str(uuid.uuid4()) + ".jpeg")
            elif(fen[j]=="k"):
                img.save("dataSepTest/k/" + str(uuid.uuid4()) + ".jpeg")
            elif(fen[j]=="P"):
                img.save("dataSepTest/pp/" + str(uuid.uuid4()) + ".jpeg")
            elif(fen[j]=="B"):
                img.save("dataSepTest/bb/" + str(uuid.uuid4()) + ".jpeg")
            elif(fen[j]=="N"):
                img.save("dataSepTest/nn/" + str(uuid.uuid4()) + ".jpeg")
            elif(fen[j]=="R"):
                img.save("dataSepTest/rr/" + str(uuid.uuid4()) + ".jpeg")
            elif(fen[j]=="Q"):
                img.save("dataSepTest/qq/" + str(uuid.uuid4()) + ".jpeg")
            elif(fen[j]=="K"):
                img.save("dataSepTest/kk/" + str(uuid.uuid4()) + ".jpeg")
            elif(fen[j]=="x"):
                img.save("dataSepTest/x/" + str(uuid.uuid4()) + ".jpeg")
            img.close()
        
    

#%%

print(str(uuid.uuid4()))
# %%
makeNewDataFormat("data/test")






#%%
tarr = os.listdir("hugeX/x")
print(len(tarr))
# %%

newTarr = np.random.choice(tarr, size=100000, replace=False) 

#%%
print(newTarr[0])

#%%
import shutil

src = "hugeX/x/"
dest = "dataSep/x/"
cnt = 0
for i in newTarr:
    cnt = cnt+1
    print(cnt)
    shutil.copy(src+i, dest+i)


# %%
