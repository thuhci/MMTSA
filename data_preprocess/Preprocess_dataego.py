import os
import sys
import subprocess
from multiprocessing import Pool
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import pickle
action2id = {'cooking on stove': 0,
 'cooking': 0,
 'cycling': 1,
 'riding elevator': 2,
 'walking down/upstairs': 3,
 'push ups': 4,
 'doing push ups': 4,
 'reading': 5,
 'washing dishes': 6,
 'working on pc': 7,
 'browsing mobile phone': 8,
 'talking with people': 9,
 'chopping': 10,
 'chopping food': 10,
 'doing sit ups': 11,
 'sit ups': 11,
 'running': 12,
 'lying down': 13,
 'eating and drinking': 14,
 'eating': 14,
 'riding escalator': 15,
 'writing': 16,
 'writting': 16,
 'brushing teeth': 17,
 'watching tv': 18,
 'walking': 19
 }

train_dic = {0:['REC_1507292576905'],
1:['REC_1507103688869','REC_1507277006736','REC_1507354660471','REC_1507353362610','REC_1507104938830'],
2:['REC_1506861601641','REC_1507022329230','REC_1507448032599'],
3:['REC_1507041645130','REC_1507043031390','REC_1507451974266'],
4: ['REC_1507103243726','REC_1507103688869','REC_1507277006736','REC_1506840511789','REC_1507085666793','REC_1507354660471'],
5: ['REC_1507538351866','REC_1507279706287','REC_1507535531191','REC_1507273499105','REC_1507193971595','REC_1507539325772'],
6: ['REC_1507193971595','REC_1507190981185'],
7: ['REC_1507357690616','REC_1506588293055','REC_1506522946460'],
8: ['REC_1507447633115','REC_1507448032599','REC_1506588293055','REC_1506522946460'],
9: ['REC_1507539325772','REC_1507535531191','REC_1507041645130','REC_1507446255664'],
10: ['REC_1507193971595','REC_1507451974266','REC_1507292576905','REC_1506858179290'],
11: [ 'REC_1507085666793','REC_1507445388999','REC_1506837019244','REC_1507102392487'],
12: [ 'REC_1506837794052','REC_1506837019244','REC_1507277006736','REC_1506840511789','REC_1507103243726'],
13:  ['REC_1507540336030','REC_1507103243726','REC_1507104938830','REC_1507085666793'],
14: ['REC_1507190981185','REC_1507193252270','REC_1506588293055','REC_1507540336030','REC_1506858179290','REC_1507535531191','REC_1507451974266','REC_1507292576905'],
15: [ 'REC_1506861601641','REC_1506862108552'],
16:  ['REC_1507539325772','REC_1507538351866','REC_1506588293055'],
17:  ['REC_1507451974266'],
18: ['REC_1507094478224'],
19:  ['REC_1507022329230','REC_1507104938830','REC_1507103243726','REC_1507354660471','REC_1507020707163','REC_1507448032599','REC_1507446255664']
}

FoldePath = "/data1/DataEgo"



"""Frame extraction"""

def vid2jpg(file_name, class_path, dst_class_path):
    if '.mp4' not in file_name:
        return
    name, ext = os.path.splitext(file_name)
    dst_directory_path = os.path.join(dst_class_path, name)

    video_file_path = os.path.join(class_path, file_name)
    try:
        if os.path.exists(dst_directory_path):
            if not os.path.exists(os.path.join(dst_directory_path, 'img_00001.jpg')):
                subprocess.call('rm -r \"{}\"'.format(dst_directory_path), shell=True)
                print('remove {}'.format(dst_directory_path))
                os.mkdir(dst_directory_path)
            else:
                print('*** convert has been done: {}'.format(dst_directory_path))
                return
        else:
            os.mkdir(dst_directory_path)
    except:
        print(dst_directory_path)
        return
    cmd = 'ffmpeg -i \"{}\" -threads 1 -r 15 -vf scale=-1:331 -q:v 0 \"{}/img_%05d.jpg\"'.format(video_file_path, dst_directory_path)
    subprocess.call(cmd, shell=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    
for datafolder in os.listdir(FoldePath):
    if '.' in datafolder:
        pass
    else:
        for file in os.listdir(os.path.join(FoldePath,datafolder)):
            if ".mp4" not in file:
                pass
            else:
                # print(file)
                # print(os.path.join(FoldePath,datafolder))
                vid2jpg(file, os.path.join(FoldePath,datafolder),os.path.join(FoldePath,"images"))


"""Sensor data produce"""
                
for datafolder in os.listdir(FoldePath):
    if '.' in datafolder or "images" in datafolder:
        pass
    else:
        local_dic = {}
        file_path = os.path.join(FoldePath,datafolder)
        with open(os.path.join(file_path,"labels.txt")) as f:
            massage = f.readlines()
        for line in massage:
            if line != "\n":
                a,b = line.rstrip("\n").strip().split(":")
                local_dic[a]= action2id[b]        
        for file in os.listdir(os.path.join(FoldePath,datafolder)):
            if "labels" in file or ".txt" not in file:
                pass
            else:
                name, _ = os.path.splitext(file)
                pd_file = pd.read_csv(os.path.join(file_path,file),header = None)
                empty_list = []
                for key_data in pd_file.iloc[:,-1]:
                    key_data = key_data.split("\t")
                    key_data.append(local_dic[key_data[-1]])
                    empty_list.append(key_data)
                np.save(os.path.join( file_path,name) ,np.array(empty_list))


for datafolder in os.listdir(FoldePath):
    if '.' in datafolder or "images" in datafolder:
        pass
    else:
        acc_npy = None
        gym_npy = None
        for file in os.listdir(os.path.join(FoldePath,datafolder)):
#             print(file)
            if ".npy" in file:
                if 'ACC' in file:
#                     print(file)
                    acc_npy = np.load(os.path.join(FoldePath,datafolder,file))
                else:
                    gym_npy = np.load(os.path.join(FoldePath,datafolder,file))
        sensor = np.c_[acc_npy[:,:3],gym_npy[:,:3],gym_npy[:,-1]]
        np.save(os.path.join(FoldePath,"images",datafolder,datafolder),sensor)
  
  
"""slide window based division"""      
empty_df = pd.DataFrame(columns=["frames_path","sensor_path","video_name","start_frame","end_frame","num_frames","label"])
df_index = 0
for image_folder in os.listdir(Frame_path):
    image_npy = np.load(os.path.join(Frame_path,image_folder,image_folder+".npy"))
    print(image_npy.shape)
    i=0
    while  i+15*15<4500:
        start = i
        end = i+15*15
        mid = (start+end)//2
#         print(start,mid,end)
        if image_npy[:,-1][start] ==  image_npy[:,-1][mid] and image_npy[:,-1][start] ==  image_npy[:,-1][end-1]:
            empty_df.loc[df_index] = [os.path.join(Frame_path,image_folder),os.path.join(Frame_path,image_folder,image_folder+".npy"),str(image_folder),start,end-1,end-1-start,image_npy[:,-1][start]]
            df_index+=1
        i+=15*10
        

"""train-test split"""
train_df = pd.DataFrame(columns=["frames_path","sensor_path","video_name","start_frame","end_frame","num_frames","label"])
test_df = pd.DataFrame(columns=["frames_path","sensor_path","video_name","start_frame","end_frame","num_frames","label"])
for label_n in range(20):

    tem_vn = train_dic[label_n]
    print(tem_vn)
    tem_df = empty_df.loc[empty_df["label"]==str(label_n)].sample(frac=1)
    
    # train_df = train_df.append(tem_df[tem_df['video_name'].isin(tem_vn)],ignore_index=True)
    # test_df = test_df.append(tem_df[~tem_df['video_name'].isin(tem_vn)],ignore_index=True)
    train_df = pd.concat([train_df, tem_df[tem_df['video_name'].isin(tem_vn)]], ignore_index=True)
    test_df = pd.concat([test_df, tem_df[~tem_df['video_name'].isin(tem_vn)]],ignore_index=True)
    

"""output"""
with open("train_dataego_file","wb") as f:
    pickle.dump(train_df, f)
    
with open("test_dataego_file","wb") as f:
    pickle.dump(test_df, f)