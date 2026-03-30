
import time
import numpy as np
import os
import av
import enum

from datacollector import Action #get action enum.
from torch.utils.data import IterableDataset
from pathlib import Path
import json
import torch
import random

class ActionDataset(IterableDataset):
    def __init__(self, session_paths : str, transform_fn):
        self.session_paths = session_paths
        self.transform = transform_fn 
        '''
        so i need to do make an iterator that keeps going forward.
        yield = “pause here, give value, continue later”
        oh yeah we can just use this right, holy shiiiiii
        def __iter__(self):
            for video in videos:
                for frame in frames:
                    yield sample

        keep yielding samples, until we run out of videos basically.. okay that makes so much sense.
        '''
    
    def __iter__(self):
        #go over every session path and make a iterator yield function.
        paths = list(self.session_paths)
        random.shuffle(paths) #shuffle paths, so we get better splits.

        for path in paths:
            path_object = Path(path)
            video_name = f"{path_object.name}.mp4"
            action_name = f"{path_object.name}.json"

            video_path = path_object / video_name
            action_path = path_object / action_name

            #load the video and action data set.abs
            action_json = None 
            container = None
            try:
                with open(action_path, 'r') as file:
                    action_json = json.load(file)
            except FileNotFoundError:
                print(f"action json not found for {path} session!")
            except json.JSONDecodeError:
                print(f"action json is corrupt for {path} session!")

            try:
                container = av.open(video_path)
            except av.AVError as e:
                print(f"video container could not be opened for session {path}")
                continue

            #get first video stream
            #video_stream = container.streams.video[0]
            frames = container.decode(video=0) #iteraotr of deocded frames!

            # json is a dict keyed by frame filename — convert values to an ordered list
            # so we can index by frame number. each value is a list of action dicts like
            # [{"action": "LEFT_CLICK", "x": 100, "y": 200}] — we just take the first one
            # as the primary action for that frame.
            action_list = list(action_json.values())

            prev_frame = None
            for i, frame in enumerate(frames):
                #if and frame,

                frame = frame.to_ndarray(format="rgb24")

                if prev_frame is None:
                    #first frame we are seeing.
                    prev_frame = frame
                    continue #skip this tuple.

                if i - 1 >= len(action_list):
                    break  # safety

                f1 = prev_frame
                f2 = frame

                # each entry is a list of action dicts, grab the first action string.
                action_str = action_list[i - 1][0]["action"]
                action = Action[action_str].value

                #now if we wnt to transofrm these fraems.
                if self.transform: #custom transform if yuo want.
                    f1 = self.transform(f1)
                    f2 = self.transform(f2)
                #else just pass raw numpy (H, W, C) uint8 — clip processor handles its own preprocessing, no need to convert here.

                action = torch.tensor(action, dtype=torch.long)

                yield (f1, f2), action
                prev_frame = frame

            container.close()

                







