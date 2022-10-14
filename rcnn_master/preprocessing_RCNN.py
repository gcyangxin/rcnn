# from __future__ import division, print_function, absolute_import
import numpy as np
from . import selectivesearch
# from . import tools
import cv2
# from . import config
import os,glob
# import random


def resize_image(in_image, new_width, new_height,padding=16, out_image=None, resize_mode=cv2.INTER_CUBIC):
    in_image=in_image.transpose(2,0,1)   
    # print(in_image.shape)
    in_image=np.pad(in_image,((0,0),(padding,padding),(padding,padding)),'mean')
    # print(in_image.shape)
    in_image=in_image.transpose(1,2,0)
    # print(in_image.shape)
    # exit()
    img = cv2.resize(in_image, (new_width, new_height),interpolation=resize_mode)
    if out_image:
        cv2.imwrite(out_image, img)
    return img


# IOU Part 1
def if_intersection(xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b):
    if_intersect = False
    if xmin_a < xmax_b <= xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_a <= xmin_b < xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_b < xmax_a <= xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    elif xmin_b <= xmin_a < xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    else:
        return if_intersect
    if if_intersect:
        x_sorted_list = sorted([xmin_a, xmax_a, xmin_b, xmax_b])
        y_sorted_list = sorted([ymin_a, ymax_a, ymin_b, ymax_b])
        x_intersect_w = x_sorted_list[2] - x_sorted_list[1]
        y_intersect_h = y_sorted_list[2] - y_sorted_list[1]
        area_inter = x_intersect_w * y_intersect_h
        return area_inter


# IOU Part 2
def IOU(ver1, vertice2):
    # vertices in four points
    vertice1 = [ver1[0], ver1[1], ver1[0]+ver1[2], ver1[1]+ver1[3]]
    area_inter = if_intersection(vertice1[0], vertice1[2], vertice1[1], vertice1[3], vertice2[0], vertice2[2], vertice2[1], vertice2[3])
    if area_inter:
        area_1 = ver1[2] * ver1[3]
        area_2 = vertice2[4] * vertice2[5]
        iou = float(area_inter) / (area_1 + area_2 - area_inter)
        return iou
    return False


# Clip Image
def clip_pic(img, rect):
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]
    x_1 = x + w
    y_1 = y + h
    # return img[x:x_1, y:y_1, :], [x, y, x_1, y_1, w, h]   
    return img[y:y_1, x:x_1, :], [x, y, x_1, y_1, w, h]


# Read in data and save data for Alexnet
# def load_train_proposals(datafile, num_clss, save_path, threshold=0.5, is_svm=False, save=False):
#     fr = open(datafile, 'r')
#     train_list = fr.readlines()
#     # random.shuffle(train_list)
#     for num, line in enumerate(train_list):

#         labels = []
#         images = []
#         tmp = line.strip().split(' ')
#         # tmp0 = image address
#         # tmp1 = label
#         # tmp2 = rectangle vertices
#         img = cv2.imread(tmp[0])
#         img_lbl, regions = selectivesearch.selective_search(
#                                img, scale=500, sigma=0.9, min_size=10)
#         candidates = set()
#         for r in regions:
#             # excluding same rectangle (with different segments)
#             if r['rect'] in candidates:
#                 continue
#             # excluding small regions
#             if r['size'] < 220:
#                 continue
#             if (r['rect'][2] * r['rect'][3]) < 500:
#                 continue
#             # resize to 227 * 227 for input
#             proposal_img, proposal_vertice = clip_pic(img, r['rect'])
#             # Delete Empty array
#             if len(proposal_img) == 0:
#                 continue
#             # Ignore things contain 0 or not C contiguous array
#             x, y, w, h = r['rect']
#             if w == 0 or h == 0:
#                 continue
#             # Check if any 0-dimension exist
#             [a, b, c] = np.shape(proposal_img)
#             if a == 0 or b == 0 or c == 0:
#                 continue
#             resized_proposal_img = resize_image(proposal_img, config.IMAGE_SIZE, config.IMAGE_SIZE)
#             candidates.add(r['rect'])
#             img_float = np.asarray(resized_proposal_img, dtype="float32")
#             images.append(img_float)
#             # IOU
#             ref_rect = tmp[2].split(',')
#             ref_rect_int = [int(i) for i in ref_rect]
#             iou_val = IOU(ref_rect_int, proposal_vertice)
#             # labels, let 0 represent default class, which is background
#             index = int(tmp[1])
#             if is_svm:
#                 if iou_val < threshold:
#                     labels.append(0)
#                 else:
#                     labels.append(index)
#             else:
#                 label = np.zeros(num_clss + 1)
#                 if iou_val < threshold:
#                     label[0] = 1
#                 else:
#                     label[index] = 1
#                 labels.append(label)
#         tools.view_bar("processing image of %s" % datafile.split('\\')[-1].strip(), num + 1, len(train_list))
#         if save:
#             np.save((os.path.join(save_path, tmp[0].split('/')[-1].split('.')[0].strip()) + '_data.npy'), [images, labels])
#     print(' ')
#     fr.close()
def load_test_proposals(threshold=0.5, is_svm=False, save=False):
    imgs_path="/root/deeplearn/archive/data/testing_images/*.jpg"
    img_paths=glob.glob(imgs_path)
    for line in img_paths:#
        num_neg=0
        labels = []
        images = []
        print(line)
        # tmp0 = image address
        # tmp1 = label
        # tmp2 = rectangle vertices
        
        img = cv2.imread(line)
        # img_t=img.copy()
        # print(line)
        img_lbl, regions = selectivesearch.selective_search(
                               img, scale=500, sigma=0.9, min_size=10)
        for r in regions:
            # excluding same rectangle (with different segments)
            # excluding small regions
            if r['size'] < 220:
                continue
            if (r['rect'][2] * r['rect'][3]) < 500:
                continue
            # resize to 227 * 227 for input

            proposal_img, proposal_vertice = clip_pic(img, r['rect'])#r['rect'] x1y1wh
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
            img_float = np.asarray(resized_proposal_img)
            # images.append(img_float)
        
def load_train_proposals(image_ids,dataframe, num_clss, save_path, threshold=0.5, is_svm=False, save=False):
    # random.shuffle(train_list)
    
    
    # for num, line in enumerate(dataframe.values):
        #image_id        xmin        ymin        xmax        ymax  x_center  y_center         w         h  classes
        #vid_4_1000  281.259045  187.035071  327.727931  223.225547  0.450434  0.539817  0.068741  0.095238     0
   
    for line in image_ids:#['vid_4_8340', 'vid_4_1900', ]
        num_neg=0
        labels = []
        images = []
        print(line)
        # tmp0 = image address
        # tmp1 = label
        # tmp2 = rectangle vertices
        img = cv2.imread(os.path.join('../input/car-object-detection/data/training_images',line+'.jpg'))
        # img_t=img.copy()
        # print(line)
        img_lbl, regions = selectivesearch.selective_search(
                               img, scale=500, sigma=0.9, min_size=10)
        gt_rect_frame=dataframe.loc[dataframe['image_id']==line]
        ref_rects_int=[]
        for kk in  gt_rect_frame.values:
        ##Index(['image_id', 'xmin', 'ymin', 'xmax', 'ymax', 
        #        'x_center', 'y_center', 'w','h', 'classes'], dtype='object')
            rect=[int(i) for i in kk[[1,2,7,8]]]#xywh
            label = np.zeros(num_clss + 2+4+4)
            label[3]=kk[5]#cx
            label[4]=kk[6]#cy
            label[5]=kk[7]#w
            label[6]=kk[8]#h
            label[2]=1 #score/iou
            #保存gt
            label[7]=kk[5]
            label[8]=kk[6]
            label[9]=kk[7]
            label[10]=kk[8]          
            
            label[1]=1#class
            labels.append(label)
            gt_img=img[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
            resized_gt_img = resize_image(gt_img, 227, 227)            
            img_float = np.asarray(resized_gt_img)
            images.append(img_float)
            ref_rects_int.append(rect)
        # cv2.rectangle(img_t,p1,p2,color=(0,255,0),thickness=1)
        for r in regions:
            # excluding same rectangle (with different segments)
            # excluding small regions
            if r['size'] < 220:
                continue
            if (r['rect'][2] * r['rect'][3]) < 500:
                continue
            # resize to 227 * 227 for input

            proposal_img, proposal_vertice = clip_pic(img, r['rect'])##r['rect'] x1y1wh
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

            # IOU
            # ref_rect =  #x1y1wh
            #[x, y, x_1, y_1, w, h]
            # p1=(proposal_vertice[:2])
            # p2=proposal_vertice[2:4]
            # print(proposal_vertice)
            for ref_rect_int in ref_rects_int:
                ## ref_rect_int:xywh
                iou_val = IOU(ref_rect_int, proposal_vertice)
            # print(iou_val)
            # labels, let 0 represent default class, which is background
                index = 1
                if is_svm:
                    if iou_val < threshold:
                        labels.append(0)
                    else:
                        labels.append(index)
                else:
                    label = np.zeros(num_clss + 2+4+4)
                    if iou_val < threshold:
                        num_neg+=1
                        if num_neg>5:                        
                            continue
                        label[0] = 1
                        label[2]=iou_val
                        
                    else:
                        label[index] = 1
                        label[2]=iou_val
                        #proposal_vertice x1, y1, x2, y2, w, h
                        label[3]=(proposal_vertice[2]-proposal_vertice[0])/2#cx
                        label[4]=(proposal_vertice[3]-proposal_vertice[1])/2#cy
                        label[5]=proposal_vertice[4]#w
                        label[6]=proposal_vertice[5]#h
                        #保存gt
                        # ref_rect_int  x y w h
                        label[7]=ref_rect_int[0]+ref_rect_int[2]/2  #cx
                        label[8]=ref_rect_int[1]+ref_rect_int[3]/2   #cy
                        label[9]=ref_rect_int[2]   #w
                        label[10]=ref_rect_int[3]  #h       
            
                    
                    
                    # cv2.rectangle(img_t,p1,p2,color=(255,0,0),thickness=1)
                        
                    labels.append(label)
                    resized_proposal_img = resize_image(proposal_img, 227, 227)
                    img_float = np.asarray(resized_proposal_img)
                    images.append(img_float)
        # tools.view_bar("processing image of %s" % datafile.split('\\')[-1].strip(), num + 1, len(train_list))
        if save:
            np.save(os.path.join(save_path, line + '_data.npy'), [images, labels])
#             return
            # cv2.imwrite(os.path.join(save_path, line[0] + '_ret.jpg'),img_t)
# load data
def load_from_npy(data_set,num=-1):
    images, labels = [], []
    if num>0:    
        data_list = [k for k in os.listdir(data_set) if '.npy' in k][:num]
    else:
        data_list = [k for k in os.listdir(data_set) if '.npy' in k]
    print('total npy files',len(data_list))
    # random.shuffle(data_list)
    for ind, d in enumerate(data_list):
        i, l = np.load(os.path.join(data_set, d),allow_pickle=True)
        # print(d,i.shape,i[0].shape)
        images.extend(i)
        labels.extend(l)        
        # tools.view_bar("load data of %s" % d, ind + 1, len(data_list))
    print('done npy')
    return images, labels
