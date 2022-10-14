import os
from random import random
import cv2
# import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, positive
from torchvision.models import alexnet,AlexNet_Weights,AlexNet
from torch.utils.data import TensorDataset,DataLoader
from rcnn_master import load_from_npy
from torch.utils.data import sampler
import random
npy_path='/root/data/Downloads/rcnn_results/train_rp/'
epoch=10

train_images,train_labels=load_from_npy(npy_path,20)

device='cpu'
#因为torch接受(b,c,w,h),所以更改维度
X_new = np.array(train_images)/255.0
X_new=X_new.transpose(0,3,1,2)
y_new = np.array(train_labels)
#筛选出1:3的正负样本
# print(X_new.shape,y_new.shape)
positive_y_inds=np.argwhere(y_new[:,1]==1).squeeze(1)
print('positive sample:',len(positive_y_inds))
negtive_y_inds=np.argwhere(y_new[:,1]==0).squeeze(1)
# print(positive_y)
# exit()

X_new=torch.from_numpy(X_new)
y_new=torch.from_numpy(y_new)

#因为torch接受(b,c,w,h),所以更改维度(b,w,h,c)
# X_new=X_new.transpose(3,1)

print('x size',X_new.shape,'y size',y_new.shape)


#加载imagenet的预训练参数
alex_weights=alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
alex_weights=alex_weights.state_dict()
#新建网络
myAlex=AlexNet(num_classes=2)
myAlex_weights=myAlex.state_dict()

#更新新建网络的权值，修改全连接层.Alex中为第4,6层
pretrained_dict = {k: v for k, v in alex_weights.items() if k in myAlex_weights and (v.shape == myAlex_weights[k].shape)}
pretrained_dict['classifier.4.weight']=torch.randn_like(myAlex_weights['classifier.4.weight'])
pretrained_dict['classifier.4.bias']=torch.zeros_like(myAlex_weights['classifier.4.bias'])
pretrained_dict['classifier.6.weight']=torch.randn_like(myAlex_weights['classifier.6.weight'])
pretrained_dict['classifier.6.bias']=torch.zeros_like(myAlex_weights['classifier.6.bias'])

# # print(pretrained_dict.keys())

myAlex.load_state_dict(pretrained_dict,strict=True)
# exit()
#只训练全连接层
for param in myAlex.parameters():
    param.requires_grad = False
for param in myAlex.classifier[4].parameters():#这里4为索引
    param.requires_grad = True
for param in myAlex.classifier[6].parameters():
    param.requires_grad = True
#fine-tuning

#冻结参数进行训练
# for p in alex.parameters():
#     p.requires_grad=False

dataset=TensorDataset(X_new,y_new)

#筛选出1:3的正负样本
negtive_y_inds=np.random.choice(negtive_y_inds,size=len(positive_y_inds)*3).tolist()
positive_y_inds=positive_y_inds.tolist()
import random
all_inds=negtive_y_inds+positive_y_inds
random.shuffle(all_inds)
#分割训练和验证集
train_simpler=[]
val_simpler=[]
for i,ind in enumerate(all_inds):
    if i %5 ==0:
        val_simpler.append(ind)
    else:
        train_simpler.append(ind)

print('train_simpler',len(train_simpler),'val_sampler',len(val_simpler))
sampler1 = sampler.SubsetRandomSampler(train_simpler)
# print(sampler1.indices)
sampler2 = sampler.SubsetRandomSampler(val_simpler)
# print(sampler1.indices)


train_loader=DataLoader(dataset=dataset,
                      batch_size=6,
                      shuffle=False,
                      num_workers=0,sampler=sampler1)
val_loader=DataLoader(dataset=dataset,
                      batch_size=2,
                      shuffle=False,
                      num_workers=0,sampler=sampler2)

optimizer=torch.optim.SGD(myAlex.parameters(),lr=0.0001)
criterion=nn.CrossEntropyLoss()

#进行训练
total=len(sampler2)
myAlex=myAlex.to(device)
for e in range(epoch):
    loss_t=0
    myAlex.train()
    for i,(x,y) in enumerate(train_loader):
        x,y=x.to(device),y.to(device)
        pred=myAlex(x)
        # print(pred.shape,y.shape)
        #pred=net(x)
        # print(pred)
        loss1 = criterion(pred,y)  # 计算损失值 
        loss_t+=loss1.cpu().item()
        optimizer.zero_grad()
        loss1.backward()                    # loss反向传播
        optimizer.step()                   # 反向传播后参数更新 
    print('epoch',e,'loss_t',loss_t/len(train_loader))
    if (e+1)%5==0:
        myAlex.eval()        
        true_x=0
        with torch.no_grad():
            for i,(img,label) in enumerate(val_loader):
                pred=myAlex(img.to(device))#nx2
                score,ind=pred.max(1)
                true_x+=torch.sum(ind.cpu()==label[:,1]).cpu().numpy().item()
        print('accuracy',true_x,total)     
        # torch.save(myAlex.state_dict(),f'myAlex_{e}.pt')


# z=0
# for e1,i in enumerate(os.listdir(path)):
# #.  z==1为了早点结束
#     if(z==1):
#         break
#     if i.startswith("428483"):
#         z += 1
#         img = cv2.imread(os.path.join(path,i))
#         ss.setBaseImage(img)
#         ss.switchToSelectiveSearchFast()
#         ssresults = ss.process()
#         imout = img.copy()
#         for e,result in enumerate(ssresults):
#         #.  同样e==50为了早点结束
#             if(e==50):
#                 break
#             if e < 2000:
#                 x,y,w,h = result
#                 timage = imout[y:y+h,x:x+w]
#                 resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
#                 img = np.expand_dims(resized, axis=0)
#                 img=torch.from_numpy(img)
#                 img=img.transpose(3,1)
#                 print(e,img.shape)
#                 out= net(img.to(torch.float32))
#                 if out[0][0] > 0.65:
#                     cv2.rectangle(imout, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
#         plt.figure()
#         plt.imshow(imout)