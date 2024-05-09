import os
import subprocess
import argparse
from tqdm import tqdm

def construct_paths(subjects, base_dir, sub_dirs):
    return [os.path.join(base_dir, sub, sub_dir) for sub in subjects for sub_dir in sub_dirs]

def vid2jpg(file_name, class_path, dst_class_path):
    if '.mp4' not in file_name:
        return
    name, ext = os.path.splitext(file_name)
    dst_directory_path = os.path.join(dst_class_path, name)

    video_file_path = os.path.join(class_path, file_name)
    try:
        if os.path.exists(dst_directory_path):
            if not os.path.exists(os.path.join(dst_directory_path, 'img_00001.jpg')):
                subprocess.call(f'rm -r "{dst_directory_path}"', shell=True)
                print('remove {}'.format(dst_directory_path))
                os.mkdir(dst_directory_path)
            else:
                print('*** convert has been done: {}'.format(dst_directory_path))
                return
        else:
            os.mkdir(dst_directory_path)
    except Exception as e:
        print(f"\n\n\n\n error {e}\n\n\n\n")
        print(dst_directory_path)
        return
    cmd = f'ffmpeg -i "{video_file_path}" -threads 1 -r 30 -vf scale=-1:331 -q:v 0 "{dst_directory_path}/img_%05d.jpg"'
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main(base_path, image_sub_path):
    video_path = os.path.join(base_path, 'Video_trimmed')
    sensor_paths = {
        'acc_phone': os.path.join(base_path, 'Sensor_trimmed/acc_phone_clip'),
        'acc_watch': os.path.join(base_path, 'Sensor_trimmed/acc_watch_clip'),
        'gyro': os.path.join(base_path, 'Sensor_trimmed/gyro_clip'),
        'orientation': os.path.join(base_path, 'Sensor_trimmed/orientation_clip')
    }
    subjects = os.listdir(video_path)
    cameras = ['cam2', 'cam1', 'cam3', 'cam4']
    scenes = ['scene1', 'scene4', 'scene2', 'scene3']
    sessions = ['session2', 'session5', 'session4', 'session1', 'session3']

    subject_video_paths = construct_paths(subjects, video_path, cameras)
    subject_video_full_paths = []
    for sub_video_path in subject_video_paths:
        for scene in scenes:
            scene_path = os.path.join(sub_video_path, scene)
            if os.path.exists(scene_path):
                for session in os.listdir(scene_path):
                    subject_video_full_paths.append(os.path.join(scene_path, session))
    
    for session in subject_video_full_paths:
        target_dir = os.path.join(image_sub_path, os.path.relpath(session, video_path))
        os.makedirs(target_dir, exist_ok=True)

        pbar = tqdm(os.listdir(session))
        for vid in pbar:
            pbar.set_description('Processing ' + session + " " + vid)
            # print(vid, session, target_dir)
            vid2jpg(vid, session, target_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert videos to images')
    parser.add_argument('--base_path', default='/data1/MMAct/', type=str, help='Base path for the MMAct dataset')
    parser.add_argument('--image_sub_path', default='/data1/MMAct/Image_subject', type=str, help='Path where images will be stored')
    args = parser.parse_args()

    main(args.base_path, args.image_sub_path)
