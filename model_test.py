import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from sklearn import svm
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
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),#torch.Size([1, 4096])
            # nn.Linear(4096, num_classes),
            # nn.LayerNorm(4096)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x




model_path='/root/data/Downloads/myAlex.pt'#全训练 lr0.0001 sgd
# image_path='/root/deeplearn/archive/data/training_images'


#加载模型参数
model_dict=torch.load(model_path,map_location='cpu')
print(model_dict['features.0.weight'][0,0,0,:])
#新建网络
myAlex=AlexNet(num_classes=2)
print(myAlex.parameters().__next__().shape)
print(myAlex.parameters().__next__()[0,0,0,:])
#修改参数这里错误导致模型预测错误！
fine_tunning_dict = {k:model_dict[k] for k, v in myAlex.state_dict().items() if k in model_dict and (v.shape == model_dict[k].shape)}

print(fine_tunning_dict['features.0.weight'][0,0,0,:])

t=myAlex.load_state_dict(fine_tunning_dict,strict=True)
print('missing key ',t)
print(myAlex.parameters().__next__().shape)
print(myAlex.parameters().__next__()[0,0,0,:])
exit()



svc_mod = joblib.load('/root/data/Downloads/svc_mod1.model')



data_x=np.load('/root/data/Downloads/rcnn_results/test_rp/vid_4_10000_data.npy',allow_pickle=True)
# print(data_x)
train_images,train_labels=data_x
# print(train_labels)
train_labels=np.array([i.tolist() for i in train_labels ])
train_images=np.array([i.tolist() for i in train_images ],dtype=np.float32)

#因为torch接受(b,c,w,h),所以更改维度
X_new = train_images/255.0
print('X_new',X_new.shape)
X_new=X_new.transpose(0,3,1,2)
print('X_new',X_new.shape)
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
    for i,(img,label) in enumerate(val_loader):
        print(img[0,0,0,0])
        pred=myAlex(img)
        print(pred[:,200])
        data_x.extend(pred.numpy())
        data_y.extend(label.numpy()[:,1])



# print(val_y)
print(svc_mod.predict(data_x))