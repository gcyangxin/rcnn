
import numpy as np
import os,cv2

def load_from_npy(data_set,num=-1):
    images, labels ,names= [], [],[]
    if num>0:    
        data_list = [k for k in os.listdir(data_set) if '.npy' in k][:num]
    else:
        data_list = [k for k in os.listdir(data_set) if '.npy' in k]
    print('total npy files',len(data_list),data_list)
    # random.shuffle(data_list)
    for ind, d in enumerate(data_list):
        i, l = np.load(os.path.join(data_set, d),allow_pickle=True)
        # print(d,i.shape,i[0].shape)
        images.extend(i)
        labels.extend(l)   
        names.extend([d]*len(images))     
        # tools.view_bar("load data of %s" % d, ind + 1, len(data_list))
    print('done npy')
    return images, labels,names


saving_path='./data_rect'
if not os.path.exists(saving_path):
    os.makedirs(saving_path)

val_npy_path='/root/data/Downloads/rcnn_results/test_rp/'

images, labels,names=load_from_npy(val_npy_path,1)
print(len(names))
n=0
print(os.path.join('/root/deeplearn/archive/data/training_images',names[0][:-9]+'.jpg'))
ori_img=cv2.imread(os.path.join('/root/deeplearn/archive/data/training_images',names[0][:-9]+'.jpg'))

for img,label,name in zip(images,labels,names):
    print(label) #label: 0 1 score pcx pcy pw ph  gcx gcy gw gh
    px1y1=(label[3]-label[5]/2,label[4]-label[6]/2)
    px2y2=(label[3]+label[5]/2,label[4]+label[6]/2)

    gx1y1=(label[7]-label[9]/2,label[8]-label[10]/2)
    gx2y2=(label[7]+label[9]/2,label[8]+label[10]/2)
    px1y1=list(map(int,px1y1))
    px2y2=list(map(int,px2y2))
    gx1y1=list(map(int,gx1y1))
    gx2y2=list(map(int,gx2y2))
    print(gx1y1,gx2y2)

    cv2.rectangle(ori_img,px1y1,px2y2,(255,0,0))
    cv2.rectangle(ori_img,gx1y1,gx2y2,(0,0,255))
    cv2.imwrite(f'/root/deeplearn/data_rect/{name[:-4]}_{n}.jpg',img)
    n+=1
#     print(name,n,np.round(label,1))
cv2.imwrite(f'/root/deeplearn/data_rect/{name[:-4]}.jpg',ori_img)
    # n+=1

