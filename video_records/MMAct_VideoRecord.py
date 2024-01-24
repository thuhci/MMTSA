from .video_record import VideoRecord

class MMAct_VideoRecord(VideoRecord):
    def __init__(self, tup):
        self._index = str(tup[0])
        self._series = tup[1]

    @property
    def untrimmed_video_name(self):
        return self._series['video_name']
    
    
    @property
    def video_path(self):
        return self._series['frames_path']
    
    @property
    def AccPhone_path(self):
        return self._series['acc_phone_path']

    @property
    def AccWatch_path(self):
        return self._series['acc_watch_path']     
    
    @property
    def Gyro_path(self):
        return self._series['gyro_path']
    
    @property
    def Orie_path(self):
        return self._series['orientation_path']
    
    @property
    def start_frame(self):
        return self._series['start_frame']

    @property
    def end_frame(self):
        return self._series['end_frame']

    @property
    def fps(self):
        return {'RGB': 30,
                'Flow': 30,
                'Spec': 60,
                'AccPhone': 100.0,
                'AccWatch': 100.0,
                'Gyro': 50.0,
                'Orie': 50.0, 
               }

    @property
    def num_frames(self):
        return {'RGB': self._series['num_frames'],
                'Flow': (self.end_frame - self.start_frame) / 2,
                'Spec': self.end_frame - self.start_frame,
                'AccPhone': self._series['num_acc_phone'],
                'AccWatch': self._series['num_acc_watch'],
                'Gyro': self._series['num_gyro'],
                'Orie': self._series['num_orie'],
               }
    @property
    def label(self):
        return self._series['label']


    @property
    def subject(self):
        return self._series['subject']


