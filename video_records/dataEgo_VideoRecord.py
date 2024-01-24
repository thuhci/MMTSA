from .video_record import VideoRecord


class DataEgo_VideoRecord(VideoRecord):
    def __init__(self, tup):
        self._index = str(tup[0])
        self._series = tup[1]

    @property
    def untrimmed_video_name(self):
        return self._series['video_name']
    
    @property
    def video_path(self):
        return self._series['frame_path']
    
    @property
    def sensor_path(self):
        return self._series['sensor_path']

    @property
    def start_frame(self):
        return self._series['start_frame']

    @property
    def end_frame(self):
        return self._series['end_frame']

#     @property
#     def fps(self):
#         return {'RGB': 15,
#                 'Flow': 30,
#                 'Spec': 60,
#                 'Sensor': 15 
#                }

    @property
    def fps(self):
        return {'RGB': 10,
                'Sensor': 10 
               }

    @property
    def num_frames(self):
        return {'RGB': self.end_frame - self.start_frame,
                'Flow': (self.end_frame - self.start_frame) / 2,
                'Spec': self.end_frame - self.start_frame,
                'Sensor': self.end_frame - self.start_frame
               }
    @property
    def label(self):
        return self._series['label']



