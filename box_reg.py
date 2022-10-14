# import os
# import cv2
# import matplotlib.pyplot as plt
from matplotlib.backend_tools import ToolXScale
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
import os,glob
import cv2
from rcnn_master.selectivesearch import selective_search
from rcnn_master.preprocessing_RCNN import clip_pic,resize_image,load_from_npy


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
            nn.MaxPool2d(kernel_size=3, stride=2),#256x6x6  torch.Size([1, 256, 6, 6])
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
        x = self.features(x)    # torch.Size([1, 256, 6, 6])    
        x2 = self.avgpool(x)    # torch.Size([1, 256, 6, 6])
        x1 = torch.flatten(x, 1) #max-pool 9216
        # print('x1.shape',x1.shape,'x2.shape',x2.shape)
        x2=torch.flatten(x2,1)
        x2 = self.classifier(x2) #分类使用 torch.Size([1, 4096])
       
        return x1,x2





model_path='/root/data/Downloads/myAlex.pt'#全训练 lr0.0001 sgd
image_path='/root/deeplearn/archive/data/testing_images'
npy_path='/root/data/Downloads/rcnn_results/train_rp/'#proposal-region  + label
val_npy_path='/root/data/Downloads/rcnn_results/test_rp/'#proposal-region  + label
test_imgs_path='/root/deeplearn/archive/data/testing_images'


# image_path='/root/deeplearn/archive/data/training_images'

#CNN模型
#加载模型参数
model_dict=torch.load(model_path,map_location='cpu')
# print(model_dict['features.0.weight'][0,0,0,:])
#新建网络
myAlex=AlexNet(num_classes=2)


#修改参数
fine_tunning_dict = {k: model_dict[k] for k, v in myAlex.state_dict().items() if k in model_dict and (v.shape == model_dict[k].shape)}
# print(fine_tunning_dict.keys())
myAlex.load_state_dict(fine_tunning_dict,strict=True)

#分类模型
svc_mod = joblib.load('/root/data/Downloads/svc_mod_1.model')

#回归模型
ridge_reg=joblib.load('/root/data/Downloads/reg_1.m')


#因为torch接受(b,c,w,h),所以更改维度
# X_new = np.array(train_images)/255.0
# X_new=X_new.transpose(0,3,1,2)
# train_labels=np.array(train_labels)
# y_new = train_labels[:,:2]
# # cxcy_new=train_labels[:,4:12]#cxcywh gtcx,gtcy,gtw,gth

# iou_mat=train_labels[:,2]#iou值

# gt_bool=((iou_mat>0.5) & (iou_mat<1)).squeeze()
# #过滤出gt
# X_new=X_new[gt_bool]
# y_new=y_new[gt_bool]
# cxcy_new=train_labels[gt_bool]
# print(X_new.shape,y_new.shape)
# print('y',y_new[0],'cxcywh gtcx,gtcy,gtw,gth',cxcy_new[0],sep='\n')

# val_x=torch.from_numpy(X_new)
# val_y=torch.from_numpy(cxcy_new)
# val_dataset=TensorDataset(val_x,val_y)
# val_loader=DataLoader(val_dataset,1,False)

# reg_loss=nn.MSELoss()

def weights(xArr, yArr, lam=1000): 
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)

    xTx = xMat.T * xMat  # 矩阵乘法  xMat.shape => (16,7)
    rxTx = xTx + np.eye(xMat.shape[1]) * lam  # 岭回归求解的括号的部分
    # 计算矩阵的值,如果值为0，说明该矩阵没有逆矩阵
    # if np.linalg.det(rxTx) == 0.0:
    #     print("This matrix cannot do inverse")
    #     return
    # xTx.I为xTx的逆矩阵
    ws = rxTx.I * xMat.T * yMat
    return ws



def test_class_run(threshold=0.5, is_svm=False, save=False):
    '测试数据输入到分类器'
    imgs_path="/root/deeplearn/archive/data/testing_images/*.jpg"
    img_paths=glob.glob(imgs_path)[:1]
    myAlex.eval()
    with torch.no_grad():
        for line in img_paths:#
            labels = []
            images = []
            # line='/root/deeplearn/archive/data/testing_images/vid_5_26620.jpg'
            line='/root/deeplearn/archive/data/training_images/vid_4_740.jpg' #train image
            print(line)
            saving_name=os.path.basename(line)
            # tmp0 = image address
            # tmp1 = label
            # tmp2 = rectangle vertices
            
            img = cv2.imread(line)
            # img_t=img.copy()
            # print(line)
            img_lbl, regions = selective_search(
                                img, scale=500, sigma=0.9, min_size=10)
            print('RegionPropasl',len(regions))
            n=0
            for r in regions:
                # excluding same rectangle (with different segments)
                # excluding small regions
                if r['size'] < 220 :
                    continue
                if (r['rect'][2] * r['rect'][3]) < 500:
                    continue
                # resize to 227 * 227 for input

                proposal_img, proposal_vertice = clip_pic(img, r['rect'])
                # x1 ,y1 ,w ,h ,x2 ,y2=proposal_vertice
                # print(proposal_vertice)
                # cv2.rectangle(img,(proposal_vertice[0],proposal_vertice[1]),(proposal_vertice[4],proposal_vertice[5]),(255,0,0))
                # cv2.imwrite(os.path.join(saving_path,saving_name[:-4]+'_tetsRP.jpg'),img)
                # exit()
                # Delete Empty array
                if len(proposal_img) == 0:
                    continue
                # Ignore things contain 0 or not C contiguous array
                x, y, w, h = r['rect']
                if w == 0 or h == 0:
                    continue
                # Check if any 0-dimension exist
                [a, b, c] = np.shape(proposal_img)
                if a == 0 or b == 0 or c == 0:
                    continue
                resized_proposal_img = resize_image(proposal_img, 227, 227)
                img_float = torch.from_numpy(resized_proposal_img)
                img_float =img_float/255.0
                img_float=torch.unsqueeze(img_float,0)
                img_float=torch.permute(img_float,(0,3,1,2))

                cxcywh=[]
                
                #proposal_vertice x1 y1  x2 y2 w h
                cxcywh.append(     
                    (proposal_vertice[2]+proposal_vertice[0])/2)#cx
                cxcywh.append(
                    (proposal_vertice[3]+proposal_vertice[1])/2)#cy
                cxcywh.append(proposal_vertice[4])#w
                cxcywh.append(proposal_vertice[5])#h  
                # cv2.rectangle(img,(proposal_vertice[0],proposal_vertice[1]),(proposal_vertice[2],proposal_vertice[3]),(255,0,0))
                # cv2.imwrite(os.path.join(saving_path,saving_name),img)

                pred_x1,pred_x2=myAlex(img_float.to(torch.float32))#Bx9216,Bx4096 对应回归和分类的特征

                classfiation=svc_mod.predict(pred_x2.numpy()).astype('int').item()

                if classfiation <1:
                    continue
                else:
                    #输出分类结果
                    # cv2.imwrite(os.path.join(saving_path,saving_name[:-4]+f'_{n}.jpg'),resized_proposal_img)
                    n+=1
                    # cv2.circle(img,(proposal_vertice[0],proposal_vertice[1]),3,(0,255,0),thickness=-1)
                    d_xywh=ridge_reg.predict(pred_x1).squeeze() 
                    print('proposal_vertice',proposal_vertice,sep='\n')
                    print('cxcywh',cxcywh,'d_xywh',d_xywh,sep='\n') 
                    pred_x=cxcywh[2]*d_xywh[0]+cxcywh[0]
                    pred_y=cxcywh[3]*d_xywh[1]+cxcywh[1]
                    pred_w=cxcywh[2]*np.exp(d_xywh[2])
                    pred_h=cxcywh[3]*np.exp(d_xywh[3]) 

                    pred_x=int(pred_x)
                    pred_y=int(pred_y)
                    pred_w=int(pred_w)
                    pred_h=int(pred_h)
                    #RP box
                    cv2.rectangle(img,(proposal_vertice[0],proposal_vertice[1]),(proposal_vertice[4],proposal_vertice[5]),(255,0,0))
                    #回归后的box
                    cv2.rectangle(img,(pred_x-pred_w//2,pred_y-pred_h//2),(pred_x+pred_w//2,pred_y+pred_h//2),(0,0,255))
        print('saving', os.path.join(saving_path,saving_name))
        cv2.imwrite(os.path.join(saving_path,saving_name[:-4]+'_2000.jpg'),img)
def train_run():
    '训练数据输入到分类回归器'
    train_images,train_labels=load_from_npy(val_npy_path)
    #因为torch接受(b,c,w,h),所以更改维度
    X_new = np.array(train_images)/255.0
    X_new=X_new.transpose(0,3,1,2)
    X_new=torch.from_numpy(X_new)
    train_labels=np.array(train_labels)
    y_new = train_labels[:,3:7]
    # iou_mat=train_labels[:,2]#iou值

    # gt_bool=((iou_mat<0.9) & (iou_mat<1.1)).squeeze()
    # X_new=X_new[gt_bool]
    # y_new=y_new[gt_bool]

    print('x_new.shape',X_new.shape)
    print('y_new.shape',y_new.shape)
    # print(train_labels)
    myAlex.eval()
    with torch.no_grad():
        n=0
        for data_x in X_new:
            datax=torch.unsqueeze(data_x,0)            
            pred_x1,pred_x2=myAlex(datax.to(torch.float32))#Bx9216,Bx4096 对应回归和分类的特征
            classfiation=svc_mod.predict(pred_x2.numpy()).astype('int').item()
            # print(classfiation,train_labels[n][1])
            # print(type(classfiation))
            # print( classfiation,train_labels[n][1])
            # exit()
            if classfiation ==0:
                continue
            else:
                img=data_x.numpy().transpose(1,2,0)*255
                print(img.shape)
                # print(img)
                # exit()
                cv2.imwrite(os.path.join(saving_path,f'{n}.jpg'),img)
            #     # n+=1
            #     print(classfiation)
            #         cv2.circle(img,(proposal_vertice[0],proposal_vertice[1]),3,(0,255,0),thickness=-1)
                # cv2.imwrite(os.path.join(saving_path,saving_name),img)
            # else:
            #     # print(classfiation.item())
            #     d_xywh=ridge_reg.predict(pred_x1).squeeze()  
            #     # print(d_xywh)
            #     cxcywh=y_new[n]
            #     print(cxcywh)
            #     pred_x=cxcywh[2]*d_xywh[0]+cxcywh[0]
            #     pred_y=cxcywh[3]*d_xywh[1]+cxcywh[1]
            #     pred_w=cxcywh[2]*np.exp(d_xywh[2])
            #     pred_h=cxcywh[3]*np.exp(d_xywh[3]) 

            #     # pred_x=int(pred_x)
            #     # pred_y=int(pred_y)
            #     # pred_w=int(pred_w)
            #     # pred_h=int(pred_h)
            #     print(pred_x,pred_y,pred_w,pred_h)
            #     print(cxcywh)
                # cv2.line(img,(proposal_vertice[0],proposal_vertice[1]),(pred_x-pred_w//2,pred_y-pred_h//2),(0,255,0),thickness=1)
                # cv2.rectangle(img,(pred_x-pred_w//2,pred_y-pred_h//2),(pred_x+pred_w//2,pred_y+pred_h//2),(0,0,255))
            # print('saving', os.path.join(saving_path,saving_name))
            # cv2.imwrite(os.path.join(saving_path,saving_name[:-4]+'_2000.jpg'),img)
            n+=1
#测试训练集分类是否正确  正确
# train_run()    
#分类结果
saving_path='./box_rect/test_class'
#回归结果
saving_path='./box_rect/test_reg'
if not os.path.exists(saving_path):
    os.makedirs(saving_path)


test_class_run()

# data_x=np.zeros_like(data_y)
# data_x[2:4,:]=1#tw th的最佳为1

# print(data_y)
# ww=weights(data_x,data_y)
# print(ww)

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



