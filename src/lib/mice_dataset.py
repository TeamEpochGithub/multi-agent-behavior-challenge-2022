import os

import cv2
import numpy as np
from torch.utils.data import Dataset


class MiceDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MouseVideoDataset(Dataset):
    """
    Reads frames from video files.
    Copied from the baseline notebook of the second round.
    """

    def __init__(
        self,
        datafolder,
        frame_number_map,
        keypoints,
        frame_skip,
        num_prev_frames,
        num_next_frames,
        frame_size=(224, 224),
        transform=None,
    ):
        """
        Initializing the dataset with images and labels
        """
        self.datafolder = datafolder
        self.transform = transform
        self.frame_number_map = frame_number_map
        self.num_prev_frames = num_prev_frames
        self.num_next_frames = num_next_frames
        self.frame_skip = frame_skip
        self.frame_size = frame_size
        self.keypoints = keypoints

        self._setup_frame_map()

    def set_transform(self, transform):
        self.transform = transform

    def _setup_frame_map(self):
        self._video_names = np.array(list(self.frame_number_map.keys()))
        # IMPORTANT: the frame number map should be sorted for self.get_video_name to work
        frame_nums = np.array([self.frame_number_map[k] for k in self._video_names])
        self._frame_numbers = frame_nums[:, 0] - 1  # start values
        assert np.all(np.diff(self._frame_numbers) > 0), "Frame number map is not sorted"

        self.length = frame_nums[-1, 1]  # last value is the total number of frames

    def get_frame_info(self, global_index):
        """Returns corresponding video name and frame number"""
        video_idx = np.searchsorted(self._frame_numbers, global_index) - 1
        frame_index = global_index - (self._frame_numbers[video_idx] + 1)
        return self._video_names[video_idx], frame_index

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name, frame_index = self.get_frame_info(idx)

        video_path = os.path.join(self.datafolder, video_name + ".avi")

        nf = self.num_next_frames + self.num_prev_frames + 1
        frames_array = np.zeros((*self.frame_size, nf), dtype=np.float32)

        if not os.path.exists(video_path):
            # raise FileNotFoundError(video_path)
            if self.transform is not None:
                frames_array = self.transform(frames_array)
            return {
                "idx": idx,
                "image": frames_array,
            }

        cap = cv2.VideoCapture(video_path)
        num_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for arridx, fnum in enumerate(
            range(
                frame_index - self.num_prev_frames * self.frame_skip,
                frame_index + self.num_next_frames * self.frame_skip + 1,
                self.frame_skip + 1,
            )
        ):
            if fnum < 0 or fnum >= num_video_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
            success, frame = cap.read()
            # print(fnum, frame_index, success)
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames_array[:, :, arridx] = frame

        if video_name in self.keypoints["sequences"]:
            bbox = self.keypoints["sequences"][video_name]["bbox"]
            if bbox.shape[0] > frame_index:
                bbox = bbox[frame_index]
                frames_array = frames_array[bbox[0] : bbox[2], bbox[1] : bbox[3]]
                # Crop the image so random crop is more useful

        if self.transform is not None:
            frames_array = self.transform(frames_array)

        return {
            "idx": idx,
            "image": frames_array,
        }
