from video_records import DataEgo_VideoRecord, CrossDataEgo_VideoRecord, MMAct_VideoRecord, mmdata_VideoRecord
import torch.utils.data as data
from PIL import Image
import os
import os.path
import pandas as pd
import numpy as np
from numpy.random import randint
import pickle


class MMTSADataSet(data.Dataset):
    def __init__(self, dataset, list_file,
                 new_length, modality, image_tmpl,
                 visual_path=None, sensor_path=None,
                 num_segments=3, transform=None,
                 mode='train', cross_dataset = False):
        self.dataset = dataset
        self.visual_path = visual_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.mode = mode
        self.cross_dataset = cross_dataset


        self._parse_list()



    def _GramianAngularField(self, series, fps = 15.0):
        image_size = series.shape[1]
        from pyts.image import GramianAngularField
        gasf = GramianAngularField(image_size=image_size, method='summation')
        sensor_gasf = gasf.fit_transform(series)
        return sensor_gasf
    
    def _normalization(self, data, scale = 255.0):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range * 255.0

    def _extract_sensor_feature(self, record, idx):

        # 确定中间秒
        centre_sec = (record.start_frame + idx) / record.fps['Sensor']
        # 左右各1s
        left_sec = centre_sec - 1.0
        right_sec = centre_sec + 1.0
        # sensor数据 (行数 x 6个channel)
        sensor_data = np.load(record.sensor_path, allow_pickle=True).astype('float')[:,:6]
        duration = sensor_data.shape[0] / float(record.fps['Sensor'])

        left_sample = int(round(left_sec * record.fps['Sensor']))
        right_sample = int(round(right_sec * record.fps['Sensor']))

        if left_sec < 0:
            samples = sensor_data[:int(round(record.fps['Sensor'] * 2.0))]

        elif right_sec > duration:
            samples = sensor_data[-int(round(record.fps['Sensor'] * 2.0)):]
        else:
            samples = sensor_data[left_sample:right_sample]

        return self._GramianAngularField(samples.transpose(), record.fps['Sensor'])
    
    def _extract_accphone_feature(self, record, idx):
        centre_sec = (record.start_frame + idx) / record.fps['AccPhone']
        left_sec = centre_sec - 1.0
        right_sec = centre_sec + 1.0
        sensor_data = np.load(record.AccPhone_path, allow_pickle=True).astype('float')[:,:3]
        duration = sensor_data.shape[0] / float(record.fps['AccPhone'])

        left_sample = int(round(left_sec * record.fps['AccPhone']))
        right_sample = int(round(right_sec * record.fps['AccPhone']))

        if left_sec < 0:
            samples = sensor_data[:int(round(record.fps['AccPhone'] * 2.0))]

        elif right_sec > duration or right_sample > sensor_data.shape[0]:
            samples = sensor_data[-int(round(record.fps['AccPhone'] * 2.0)):]
        else:
            samples = sensor_data[left_sample:right_sample]

        return self._GramianAngularField(samples.transpose(), record.fps['AccPhone'])


    def _extract_accwatch_feature(self, record, idx):
        centre_sec = (record.start_frame + idx) / record.fps['AccWatch']
        left_sec = centre_sec - 1.0
        right_sec = centre_sec + 1.0
        sensor_data = np.load(record.AccWatch_path, allow_pickle=True).astype('float')[:,:3]
        duration = sensor_data.shape[0] / float(record.fps['AccWatch'])

        left_sample = int(round(left_sec * record.fps['AccWatch']))
        right_sample = int(round(right_sec * record.fps['AccWatch']))

        if left_sec < 0:
            samples = sensor_data[:int(round(record.fps['AccWatch'] * 2.0))]

        elif right_sec > duration or right_sample > sensor_data.shape[0]:
            samples = sensor_data[-int(round(record.fps['AccWatch'] * 2.0)):]
        else:
            samples = sensor_data[left_sample:right_sample]

        return self._GramianAngularField(samples.transpose(), record.fps['AccWatch'])
    
    def _extract_gyro_feature(self, record, idx):
        centre_sec = (record.start_frame + idx) / record.fps['Gyro']
        left_sec = centre_sec - 1.0
        right_sec = centre_sec + 1.0
        sensor_data = np.load(record.Gyro_path, allow_pickle=True).astype('float')[:,:3]
        duration = sensor_data.shape[0] / float(record.fps['Gyro'])

        left_sample = int(round(left_sec * record.fps['Gyro']))
        right_sample = int(round(right_sec * record.fps['Gyro']))

        if left_sec < 0:
            samples = sensor_data[:int(round(record.fps['Gyro'] * 2.0))]

        elif right_sec > duration or right_sample > sensor_data.shape[0]:
            samples = sensor_data[-int(round(record.fps['Gyro'] * 2.0)):]
        else:
            samples = sensor_data[left_sample:right_sample]

        return self._GramianAngularField(samples.transpose(), record.fps['Gyro'])
    
    def _extract_orie_feature(self, record, idx):
        centre_sec = (record.start_frame + idx) / record.fps['Orie']
        left_sec = centre_sec - 1.0
        right_sec = centre_sec + 1.0
        sensor_data = np.load(record.Orie_path, allow_pickle=True).astype('float')[:,:3]
        duration = sensor_data.shape[0] / float(record.fps['Orie'])

        left_sample = int(round(left_sec * record.fps['Orie']))
        right_sample = int(round(right_sec * record.fps['Orie']))

        if left_sec < 0:
            samples = sensor_data[:int(round(record.fps['Orie'] * 2.0))]

        elif right_sec > duration or right_sample > sensor_data.shape[0]:
            samples = sensor_data[-int(round(record.fps['Orie'] * 2.0)):]
        else:
            samples = sensor_data[left_sample:right_sample]

        return self._GramianAngularField(samples.transpose(), record.fps['Orie'])


    def _load_data(self, modality, record, idx):
        if self.dataset == 'MMAct':
            video_path = record.video_path
        else:
            video_path = os.path.join(self.visual_path, record.untrimmed_video_name)
            
        if modality == 'RGB':
            idx_untrimmed = record.start_frame + idx
            if idx_untrimmed==0:
                idx_untrimmed += 1
            return [Image.open(os.path.join(video_path, self.image_tmpl[modality].format(idx_untrimmed))).convert('RGB')]
        elif modality =="Sensor":
            sens = self._extract_sensor_feature(record, idx)
            return [Image.fromarray(self._normalization(single_channel)).convert('L') for single_channel in sens]
        elif modality =="AccPhone":
            sens = self._extract_accphone_feature(record, idx)
            return [Image.fromarray(self._normalization(single_channel)).convert('L') for single_channel in sens]
        elif modality =="AccWatch":
            sens = self._extract_accwatch_feature(record, idx)
            return [Image.fromarray(self._normalization(single_channel)).convert('L') for single_channel in sens]
        elif modality =="Gyro":
            sens = self._extract_gyro_feature(record, idx)
            return [Image.fromarray(self._normalization(single_channel)).convert('L') for single_channel in sens]
        elif modality =="Orie":
            sens = self._extract_orie_feature(record, idx)
            return [Image.fromarray(self._normalization(single_channel)).convert('L') for single_channel in sens]
                                  

    def _parse_list(self):
        if self.dataset == 'dataEgo':
            if self.cross_dataset == False:
                self.video_list = [DataEgo_VideoRecord(tup) for tup in self.list_file.iterrows()]
            else:
                self.video_list = [CrossDataEgo_VideoRecord(tup) for tup in self.list_file.iterrows()]
        elif self.dataset == 'MMAct':
            self.video_list = [MMAct_VideoRecord(tup) for tup in self.list_file.iterrows()]
        elif self.dataset == 'mmdata':
            self.video_list = [mmdata_VideoRecord(tup) for tup in self.list_file.iterrows()]

    def _sample_indices(self, record, modality):
        """
        :param record: VideoRecord
        :return: list
        """
        average_duration = (record.num_frames[modality] - self.new_length[modality] + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_val_indices(self, record, modality):
        if record.num_frames[modality] > self.num_segments + self.new_length[modality] - 1:
            tick = (record.num_frames[modality] - self.new_length[modality] + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def __getitem__(self, index):
        input = {}
        record = self.video_list[index]
        for m in self.modality:
            if self.mode == 'train':
                segment_indices = self._sample_indices(record, m)
            else:
                segment_indices = self._get_val_indices(record, m)


            if m != 'RGB' and self.mode == 'train':
                np.random.shuffle(segment_indices)

            img, label = self.get(m, record, segment_indices)
            input[m] = img
            
#         print(index, input['RGB'].shape, input['Sensor'].shape)
        return input, label

    def get(self, modality, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length[modality]):
                seg_imgs = self._load_data(modality, record, p)
                images.extend(seg_imgs)
                if p < record.num_frames[modality]:
                    p += 1
        process_data = self.transform[modality](images)

        return process_data, int(record.label)

    def __len__(self):
        return len(self.video_list)
