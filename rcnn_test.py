import os
import cv2
# import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, positive
from torchvision.models import AlexNet
from torch.utils.data import TensorDataset,DataLoader,Sampler
from rcnn_master import load_from_npy,load_test_proposals
from torch.utils.data import sampler


model_path='/root/data/Downloads/myAlex_99.pt'#全训练 lr0.0001 sgd 
model_path='/root/data/Downloads/myAlex_79.pt'#全训练 lr0.0001 sgd gt +iou0.56
image_path='/root/deeplearn/archive/data/testing_images'
# image_path='/root/deeplearn/archive/data/training_images'
model_dict=torch.load(model_path,map_location='cpu')
#新建网络
myAlex=AlexNet(num_classes=2)
myAlex.load_state_dict(model_dict,strict=True)
img_list=[ i for i in  os.listdir(image_path) if '.jpg' in i][:20]
save_path='classfiation_all_sgd_wh'
if not os.path.exists(save_path):
    os.makedirs(save_path)
n=0    
for  line in img_list:
    print(line)
    test_images=load_test_proposals(line,image_path)
    print('test_images',test_images.shape)

    # X_new=torch.from_numpy(X_new)
    # y_new=torch.from_numpy(y_new)
    test_images=test_images.transpose(0,3,1,2)#b,w,h,c
    test_images=torch.from_numpy(test_images)/255.0
    # test_images=test_images.transpose(3,1)

    test_dataset=TensorDataset(test_images)
    test_loader=DataLoader(test_dataset,1,False)
    myAlex.eval()
    with torch.no_grad():
        for i,img in enumerate(test_loader):
            img=img[0]
            pred=myAlex(img.to(torch.float32))
            score,ind=torch.max(pred,1)
            if ind.numpy().item()>0:
                print('1111')
                img_t=img.squeeze().numpy()#cwh
                print(img_t.shape)
                img_t=img_t.transpose(1,2,0)*255
                cv2.imwrite(f'{save_path}/{n}.jpg',img_t)
                n+=1
            
            




