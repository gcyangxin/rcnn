# import os
# import cv2
# import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
# from torchvision.models import AlexNet
from torch.utils.data import TensorDataset,DataLoader,sampler
from rcnn_master import load_from_npy,load_test_proposals
# from torch.utils.data import sampler
from sklearn import svm
from sklearn.metrics import classification_report
import joblib
import os

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),#fc5
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),#fc6 . features from  fc7 generalize worse than features from fc6,所以可以移除fc7，减小参宿量4096*num_classes
            nn.ReLU(inplace=True),#torch.Size([1, 4096])
            # nn.Linear(4096, num_classes),#fc7
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


model_path='/root/data/Downloads/myAlex.pt'#全训练 lr0.0001 sgd
image_path='/root/deeplearn/archive/data/testing_images'
npy_path='/root/data/Downloads/rcnn_results/train_rp/'#proposal-region  + label
val_npy_path='/root/data/Downloads/rcnn_results/test_rp/'#proposal-region  + label
# image_path='/root/deeplearn/archive/data/training_images'


#加载模型参数
model_dict=torch.load(model_path,map_location='cpu')
# print(model_dict['features.0.weight'][0,0,0,:])
#新建网络
myAlex=AlexNet(num_classes=2)


#修改参数
fine_tunning_dict = {k: model_dict[k] for k, v in myAlex.state_dict().items() if k in model_dict and (v.shape == model_dict[k].shape)}
# print(fine_tunning_dict.keys())
myAlex.load_state_dict(fine_tunning_dict,strict=True)
# print(myAlex.parameters().__next__()[0,0,0,:])

#若存在svm模型则直接调用，否则进行训练
if os.path.exists('/root/data/Downloads/svc_mod1.model'):
    print('load svc model...')
    svc_mod = joblib.load('/root/data/Downloads/svc_mod1.model')
    # import pickle
    # with open('/root/data/Downloads/svc_mod1.pkl','rb') as f:
    #     svc_mod=pickle.load(f)
    # print(dir(svc_mod))
    # print(svc_mod.support_vectors_)
    # exit()
else:
    #加载所有训练的RegionProposal
    print(yx)
    train_images,train_labels=load_from_npy(npy_path,200)
    #
    X_new = np.array(train_images)
    print(X_new.shape)
    X_new=X_new.transpose(0,3,1,2)/255.0
    y_new = np.array(train_labels)

    #筛选出1:3的正负样本
    positive_y_inds=np.argwhere(y_new[:,1]==1).squeeze(1)
    negtive_y_inds=np.argwhere(y_new[:,1]==0).squeeze(1)
    negtive_y_inds=np.random.choice(negtive_y_inds,size=len(positive_y_inds)*2).tolist()
    positive_y_inds=positive_y_inds.tolist()
    sampler1 = sampler.SubsetRandomSampler(negtive_y_inds+positive_y_inds)

    X_new=torch.from_numpy(X_new)
    y_new=torch.from_numpy(y_new)
    #torch接受(b,c,h,w)
    train_dataset=TensorDataset(X_new,y_new)
    train_loader=DataLoader(train_dataset,10,False,sampler=sampler1)

    #训练svc分类器
    svc_mod=svm.SVC(kernel='linear')
    myAlex.eval()
    with torch.no_grad():
        data_x=np.zeros((0,4096))
        data_y=np.zeros(0)
        for i,(img,label) in enumerate(train_loader):#label (1x2) onehot。属于该类的概率
            # img=img[0]
            pred=myAlex(img.to(torch.float32))#[n,4096] n=batch,每个RP
            #LayerNorm

            # print(pred.shape,label)
            data_x=np.vstack((pred.numpy(),data_x))
            data_y=np.hstack((label.numpy()[:,1],data_y))#做二分类，把除了本身的赋为0
            # data_y=data_y[:,1]
    print('training shape')
    print(data_x.shape)
    print(data_y)
    svc_mod=svc_mod.fit(data_x,data_y)
    joblib.dump(svc_mod, 'svc_mod.model')       
    # del data_x,data_y,X_new,train_loader

#加载验证的RP
train_images,train_labels=load_from_npy(val_npy_path)

#因为torch接受(b,c,w,h),所以更改维度
X_new = np.array(train_images)/255.0
X_new=X_new.transpose(0,3,1,2)
y_new = np.array(train_labels)[:,:2]

# val_x=val_x.transpose(0,3,1,2)
val_y = np.array(y_new)[:,:2]
# print(val_y)
val_x=torch.from_numpy(X_new)
val_y=torch.from_numpy(val_y)
val_dataset=TensorDataset(val_x,val_y)
val_loader=DataLoader(val_dataset,5,False)

myAlex.eval()
with torch.no_grad():
    data_x=[]
    data_y=[]
    for i,(img,label) in enumerate(val_loader):#label (1x2) onehot。属于该类的概率
        # img=img[0]
        pred=myAlex(img.to(torch.float32))#[n,4096] n=batch,每个RP
        # print(img.shape,label.shape)
        # exit()
        # print(pred.shape,label)
        # pred=normalize(pred,dim=1)#normalize函数 dim=1,整体除以了第一行的范数
        data_x.extend(pred.numpy())
        data_y.extend(label.numpy()[:,1])
        # print(label.numpy()[:,1])
        # x_data=pred.numpy()
        # data_y=np.hstack((label.numpy()[:,1],data_y))#做二分类，把除了本身的赋为0
        # data_y=data_y[:,1]

data_x=np.array(data_x)
print('data_x shape',data_x.shape)
data_y=np.array(data_y)
y_pred=svc_mod.predict(data_x)
print('score',svc_mod.score(data_x,data_y))
print('y_pred',data_x.shape)
print(y_pred)
print('label')
print(data_y)

# print(classification_report(data_y,y_pred))
# svc_mod.predict()
        # score,ind=torch.max(pred,1)
        # if ind.numpy().item()>0 and score.numpy().item() >0.5:
        #     print('1111')
        #     img_t=img.squeeze().numpy()#cwh
        #     print(img_t.shape)
        #     img_t=img_t.transpose(1,2,0)*255
        #     cv2.imwrite(f'{save_path}/{n}.jpg',img_t)
        #     n+=1



