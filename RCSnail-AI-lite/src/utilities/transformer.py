import numpy as np
import pandas as pd
import cv2
from PIL import Image

from src.learning.training.collector import Collector
from src.utilities.memory_maker import MemoryMaker


class Transformer:
    def __init__(self, config, memory_tuple=None):
        self.resolution = (config.frame_width, config.frame_height)
        self.__labels = Collector()

    def cut_wide_and_normalize_video_shifted(self, frames_list):
        resized_frames = np.zeros((len(frames_list) - 1, self.resolution[1], self.resolution[0], 3), dtype=np.float32)
        frame_height = frames_list[0].shape[0]
        for i in range(0, resized_frames.shape[0]):
            resized_frames[i] = frames_list[i][(frame_height - self.resolution[1]):, :, :].astype(np.float32)
        resized_frames /= 255
        return resized_frames

    def session_frame_wide(self, frame, memory_list):
        resized = frame[(frame.shape[0] - self.resolution[1]):, :, :].astype(np.float32)
        resized /= 255
        
    def session_expert_action(self, expert_action):
        df = pd.DataFrame.from_records([expert_action], columns=expert_action.keys())[expert_action.keys()]
        return self.__labels.collect_df_columns(df, self.__labels.diff_columns()).to_numpy()[0]
