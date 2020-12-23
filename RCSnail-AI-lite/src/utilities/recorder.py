import os
import logging
import datetime

import numpy as np
import pandas as pd
import cv2


class Recorder:
    def __init__(self, config):
        self.storage_full_path = self.__get_training_file_name(config.path_to_training)
        self.resolution = (config.recording_width, config.recording_height)
        self.fps = config.recording_fps

        self.frames = []
        self.telemetry = []
        self.expert_actions = []
        self.predictions = []

    def __get_training_file_name(self, path_to_training):
        date = datetime.datetime.today().strftime("%Y_%m_%d")
        files_from_same_date = list(filter(lambda file: date in file, os.listdir(path_to_training)))
        return '{}{}_i{}'.format(path_to_training, date, str(int(len(files_from_same_date) / 2 + 1)))

    def record_full(self, frame, telemetry, expert_actions, predictions):
        if telemetry is not None and frame is not None and expert_actions is not None:
            self.frames.append(frame)
            self.telemetry.append(telemetry)
            self.expert_actions.append(expert_actions)
            self.predictions.append(predictions)
            return 1
        return 0

    def get_current_data(self):
        return self.frames, self.telemetry, self.expert_actions

    def save_session_with_expert(self):
        session_length = len(self.telemetry)
        assert session_length == len(self.frames) == len(self.expert_actions), "Stored actions are not of same length."

        if session_length <= 0:
            logging.info("Nothing to record, closing.")
            return

        logging.info("Number of training instances to be saved: " + str(session_length))

        out = cv2.VideoWriter(self.storage_full_path + ".avi",
                              cv2.VideoWriter_fourcc(*'DIVX'),
                              self.fps,
                              self.resolution)

        for i in range(session_length):
            out.write(self.frames[i].astype(np.uint8))
        out.release()

        df_telem = pd.DataFrame(self.telemetry)
        df_expert = pd.DataFrame(self.expert_actions)
        df = pd.concat([df_telem, df_expert], axis=1)
        df.to_csv(self.storage_full_path + '.csv')

        logging.info("Telemetry, expert, and video saved successfully.")
