import pandas as pd
# from rcnn_master import load_train_proposals,load_test_proposals

def get_label():
    '''#the center of proposal Pi’s bounding box together with Pi’s width and height in pixels.'''
    img_h, img_w, num_channels = (380, 676, 3)
    df = pd.read_csv('/root/deeplearn/archive/data/train_solution_bounding_boxes (1).csv')
    df.rename(columns={'image':'image_id'}, inplace=True)
    df['image_id'] = df['image_id'].apply(lambda x: x.split('.')[0])
    df['x_center'] = (df['xmin'] + df['xmax'])/2
    df['y_center'] = (df['ymin'] + df['ymax'])/2
    df['w'] = df['xmax'] - df['xmin']
    df['h'] = df['ymax'] - df['ymin']
    df['classes'] = 1
    # df['x_center'] = df['x_center']
    # df['w'] = df['w']
    # df['y_center'] = df['y_center']
    # df['h'] = df['h']
    index = sorted(list(set(df.image_id)))
    
    # print(df.columns)
    print(df.head(1))
    # image = random.choice(index)
    return df,index
test_image_path='/root/deeplearn/archive/data/testing_images'
dataframe,index=get_label()
save_path='./train_rp'

# load_train_proposals(index,dataframe,1,save_path,save=False)
# load_test_proposals(test_image_path,'./')

# image = index[0]
# img = cv2.imread(f'/root/deeplearn/archive/data/training_images/{image}.jpg')

# img_list_path='/root/deeplearn/archive/data/training_name.txt'

# load_train_proposals(img_list_path,1,'./rp_ret')

