import os
import subprocess
import argparse
from tqdm import tqdm
import pandas as pd
import sys


def main(base_path, image_sub_path, div = "subject"):
    def find_leaf_directories(root_dir):
        leaf_dirs = []
        for root, dirs, files in os.walk(root_dir):
            if not dirs: 
                leaf_dirs.append(root)
        return leaf_dirs

    img_folders = find_leaf_directories(image_sub_path)
    
    sensor_paths = {
        'acc_phone': os.path.join(base_path, 'Sensor_trimmed/acc_phone_clip/'),
        'acc_watch': os.path.join(base_path, 'Sensor_trimmed/acc_watch_clip/'),
        'gyro': os.path.join(base_path, 'Sensor_trimmed/gyro_clip/'),
        'orientation': os.path.join(base_path, 'Sensor_trimmed/orientation_clip/')
    }

        
        
    # pd.read_csv("/data1/MMAct/Sensor_trimmed/acc_phone_clip/subject19/scene4/session2/pulling.csv",header=None).values[:,1:]
        
    fails = []
    
    for video_file in img_folders:
        # print(video_file)
        try:
            video_item = video_file.split('/')
            # print(sensor_paths["acc_phone"]+video_item[-5]+'/'+video_item[-3]+'/'+video_item[-2]+'/'+video_item[-1]+'.csv')
            file1 = pd.read_csv(sensor_paths["acc_phone"]+video_item[-5]+'/'+video_item[-3]+'/'+video_item[-2]+'/'+video_item[-1]+'.csv',header=None).values[:,1:]
            file2 = pd.read_csv(sensor_paths["acc_watch"]+video_item[-5]+'/'+video_item[-3]+'/'+video_item[-2]+'/'+video_item[-1]+'.csv',header=None).values[:,1:]
            file3 = pd.read_csv(sensor_paths["gyro"]+video_item[-5]+'/'+video_item[-3]+'/'+video_item[-2]+'/'+video_item[-1]+'.csv',header=None).values[:,1:]
            file4 = pd.read_csv(sensor_paths["orientation"]+video_item[-5]+'/'+video_item[-3]+'/'+video_item[-2]+'/'+video_item[-1]+'.csv',header=None).values[:,1:]

            if not os.path.exists(sensor_paths["acc_phone"]+video_item[4]+'/'+video_item[6]+'/'+video_item[7]+'/'+video_item[8]+'.npy'):
                np.save(sensor_paths["acc_phone"]+video_item[4]+'/'+video_item[6]+'/'+video_item[7]+'/'+video_item[8]+'.npy',file1)
            if not os.path.exists(sensor_paths["acc_watch"]+video_item[4]+'/'+video_item[6]+'/'+video_item[7]+'/'+video_item[8]+'.npy'):
                np.save(sensor_paths["acc_watch"]+video_item[4]+'/'+video_item[6]+'/'+video_item[7]+'/'+video_item[8]+'.npy',file2)
            if not os.path.exists(sensor_paths["gyro"]+video_item[4]+'/'+video_item[6]+'/'+video_item[7]+'/'+video_item[8]+'.npy'):
                np.save(sensor_paths["gyro"]+video_item[4]+'/'+video_item[6]+'/'+video_item[7]+'/'+video_item[8]+'.npy',file3)
            if not os.path.exists(sensor_paths["orientation"]+video_item[4]+'/'+video_item[6]+'/'+video_item[7]+'/'+video_item[8]+'.npy'):
                np.save(sensor_paths["orientation"]+video_item[4]+'/'+video_item[6]+'/'+video_item[7]+'/'+video_item[8]+'.npy',file4)
            
        except:
            fails.append(video_file)
        
    # clean img folders
    for fail in fails:
        img_folders.remove(fail)
        
    print(len(fails), len(img_folders))
    
    if div == "subject":
        empty_df = pd.DataFrame(columns=["frames_path","acc_phone_path","acc_watch_path","gyro_path","orientation_path","video_name","start_frame","end_frame","num_frames","num_acc_phone","num_acc_watch","num_gyro","num_orie","label","subject"])
        df_index = 0

        for video_fil in img_folders:
            video_item = video_fil.split('/')
            accPhone_fil = sensor_paths["acc_phone"]+video_item[-5]+'/'+video_item[-3]+'/'+video_item[-2]+'/'+video_item[-1]+'.npy'
            accWatch_fil = sensor_paths["acc_watch"]+video_item[-5]+'/'+video_item[-3]+'/'+video_item[-2]+'/'+video_item[-1]+'.npy'
            gyro_fil = sensor_paths["gyro"]+video_item[-5]+'/'+video_item[-3]+'/'+video_item[-2]+'/'+video_item[-1]+'.npy'
            orie_fil = sensor_paths["orientation"]+video_item[-5]+'/'+video_item[-3]+'/'+video_item[-2]+'/'+video_item[-1]+'.npy'
            frame_len = len(os.listdir(video_fil))
            acc_w_len = np.load(accWatch_fil,allow_pickle=True).shape[0]
            acc_p_len = np.load(accPhone_fil,allow_pickle=True).shape[0]
            gyro_len = np.load(gyro_fil,allow_pickle=True).shape[0]
            orie_len = np.load(orie_fil,allow_pickle=True).shape[0]
            
            if video_item[-3] == 'scene3' and (video_item[-1] == 'using_phone' or video_item[-1] == 'talking_on_phone'):
                if video_item[-1] == 'using_phone':
                    empty_df.loc[df_index] = [video_fil, accPhone_fil,accWatch_fil,gyro_fil,orie_fil, video_item[-1], 1, frame_len-1, frame_len-1, acc_p_len, acc_w_len, gyro_len, orie_len, Activity_id['using_phone_desk'], video_item[-5]]
                elif video_item[-1] == 'talking_on_phone':
                    empty_df.loc[df_index] = [video_fil, accPhone_fil,accWatch_fil,gyro_fil,orie_fil, video_item[-1], 1, frame_len-1, frame_len-1, acc_p_len, acc_w_len, gyro_len, orie_len, Activity_id['talking_on_phone_desk'], video_item[-5]]
            else:
                empty_df.loc[df_index] = [video_fil, accPhone_fil,accWatch_fil,gyro_fil,orie_fil, video_item[-1], 1, frame_len-1, frame_len-1, acc_p_len, acc_w_len, gyro_len, orie_len, Activity_id[video_item[-1]], video_item[-5]]

                
            df_index+=1
        
        
        train_df = pd.DataFrame(columns=["frames_path","acc_phone_path","acc_watch_path","gyro_path","orientation_path","video_name","start_frame","end_frame","num_frames","num_acc_phone","num_acc_watch","num_gyro","num_orie","label","subject"])
        test_df = pd.DataFrame(columns=["frames_path","acc_phone_path","acc_watch_path","gyro_path","orientation_path","video_name","start_frame","end_frame","num_frames","num_acc_phone","num_acc_watch","num_gyro","num_orie","label","subject"])

        train_df = train_df.append(empty_df.loc[~empty_df["subject"].isin(['subject17', 'subject18','subject19','subject20'])],ignore_index=True)
        test_df = test_df.append(empty_df.loc[empty_df["subject"].isin(['subject17', 'subject18','subject19','subject20'])],ignore_index=True)    
            
        with open("train_mmact_file","wb") as f:
            pickle.dump(train_df, f)
        with open("test_mmact_file","wb") as f:
            pickle.dump(test_df, f)
            
    elif div == "session":
        empty_df = pd.DataFrame(columns=["frames_path","acc_phone_path","acc_watch_path","gyro_path","orientation_path","video_name","start_frame","end_frame","num_frames","num_acc_phone","num_acc_watch","num_gyro","num_orie","label","subject","session"])
        df_index = 0

        for video_fil in Video_folder:
            video_item = video_fil.split('/')
            accPhone_fil = AccPhone_Sensor_path+video_item[-5]+'/'+video_item[-3]+'/'+video_item[-2]+'/'+video_item[-1]+'.npy'
            accWatch_fil = AccWatch_Sensor_path+video_item[-5]+'/'+video_item[-3]+'/'+video_item[-2]+'/'+video_item[-1]+'.npy'
            gyro_fil = Gyro_Sensor_path+video_item[-5]+'/'+video_item[-3]+'/'+video_item[-2]+'/'+video_item[-1]+'.npy'
            orie_fil = Orie_Sensor_path+video_item[-5]+'/'+video_item[-3]+'/'+video_item[-2]+'/'+video_item[-1]+'.npy'
            frame_len = len(os.listdir(video_fil))
            acc_w_len = np.load(accWatch_fil,allow_pickle=True).shape[0]
            acc_p_len = np.load(accPhone_fil,allow_pickle=True).shape[0]
            gyro_len = np.load(gyro_fil,allow_pickle=True).shape[0]
            orie_len = np.load(orie_fil,allow_pickle=True).shape[0]
            
            if video_item[-3] == 'scene3' and (video_item[-1] == 'using_phone' or video_item[-1] == 'talking_on_phone'):
                if video_item[-1] == 'using_phone':
                    empty_df.loc[df_index] = [video_fil, accPhone_fil, accWatch_fil, gyro_fil, orie_fil, video_item[-1], 1, frame_len-1, frame_len-1, acc_p_len, acc_w_len, gyro_len, orie_len, Activity_id['using_phone_desk'], video_item[-5], video_item[-2]]
                elif video_item[-1] == 'talking_on_phone':
                    empty_df.loc[df_index] = [video_fil, accPhone_fil, accWatch_fil, gyro_fil, orie_fil, video_item[-1], 1, frame_len-1, frame_len-1, acc_p_len, acc_w_len, gyro_len, orie_len, Activity_id['talking_on_phone_desk'], video_item[-5], video_item[-2]]
            else:
                empty_df.loc[df_index] = [video_fil, accPhone_fil,accWatch_fil,gyro_fil,orie_fil, video_item[-1], 1, frame_len-1, frame_len-1, acc_p_len, acc_w_len, gyro_len, orie_len, Activity_id[video_item[-1]], video_item[-5], video_item[-2]]

                
            df_index+=1
            
        train_df = pd.DataFrame(columns=["frames_path","acc_phone_path","acc_watch_path","gyro_path","orientation_path","video_name","start_frame","end_frame","num_frames","num_acc_phone","num_acc_watch","num_gyro","num_orie","label","subject","session"])
        test_df = pd.DataFrame(columns=["frames_path","acc_phone_path","acc_watch_path","gyro_path","orientation_path","video_name","start_frame","end_frame","num_frames","num_acc_phone","num_acc_watch","num_gyro","num_orie","label","subject","session"])

        train_df = train_df.append(empty_df.loc[empty_df["session"].isin(['session1', 'session2','session3','session4'])],ignore_index=True)
        test_df = test_df.append(empty_df.loc[empty_df["session"].isin(['session5'])],ignore_index=True)
        with open("train_mmact_session_file","wb") as f:
            pickle.dump(train_df, f)
        with open("test_mmact_session_file","wb") as f:
            pickle.dump(test_df, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Spilt')
    parser.add_argument('--base_path', default='/data1/MMAct/', type=str, help='Base path for the MMAct dataset')
    parser.add_argument('--image_sub_path', default='/data1/MMAct/Image_subject', type=str, help='Path where images will be stored')
    parser.add_argument('--div', type=str, choices=['subject', 'session'], default='subject',
                    help='Division of the dataset: "subject" divides by subjects, "session" divides by sessions.')


    args = parser.parse_args()

    main(args.base_path, args.image_sub_path, args.div)
