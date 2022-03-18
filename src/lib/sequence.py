import numpy as np


class Sequence:
    def __init__(self, name, seq_array: np.ndarray, chasing_labels=None, light_label=None):
        self.name = name
        # 72 = (num mice = 3) x (body parts = 12) x (x, y = 2)
        flat = seq_array.reshape((1800, 72))
        self.frames = flat
        self.light_label = light_label
        self.chasing_labels = chasing_labels

    def get_mouse(self, mouse_idx, frame_idx):
        """
        get first, second or third mouse
        :param mouse_idx: index of mouse
        :param frame_idx: from which frame
        :return: np array with 24 numbers (nose_x, nose_y, left_ear_x, ...)
        """
        return Sequence.mouse_from_frame(self.frames[frame_idx], mouse_idx)

    def set_mouse(self, mouse_idx, frame_idx, mouse: np.ndarray):
        if mouse.shape != (24,):
            raise ValueError(f"Incorrect mouse shape, expected (24, ), got {mouse.shape}")
        self.frames[frame_idx][mouse_idx * 24 : (mouse_idx + 1) * 24] = mouse

    def convert_to_vame_frame(self, single_mouse_embedding: bool) -> np.ndarray:
        """
        Converts the sequence instance into a vame-compatible 2d numpy array
        :param single_mouse_embedding: true to isolate individual mice,
        false to convert them all at the same time.
        :return: the numpy array
        """
        # If true, returns (5400, 36) array, otherwise returns (1800, 108)
        if not single_mouse_embedding:
            # For now not implemented
            return np.zeros(0)
        num_frames = 1800
        num_mice = 3
        # 3 mice * 1800 frames = 5400, 3 cols * 12 keypoints = 36
        data_arr = np.zeros((5400, 36))
        for mouse_idx in range(num_mice):
            for f_num in range(num_frames):
                # Indexing data_arr
                data_idx = f_num + mouse_idx * num_frames
                # Getting keypoints
                mouse_kpts = self.get_mouse(mouse_idx, f_num)
                likelihood_vals = list(range(2, 24 + 1, 2))
                row = np.insert(mouse_kpts, likelihood_vals, 1.0)
                data_arr[data_idx, :] = row
        return data_arr

    @staticmethod
    def mouse_from_frame(frame: np.ndarray, mouse_idx) -> np.ndarray:
        return frame[mouse_idx * 24 : (mouse_idx + 1) * 24]

    @staticmethod
    def name_mouse(mouse: np.ndarray) -> dict:
        """
        generates dict with (x, y) for all named mouse body parts
        :param mouse: as 24 numbers
        :return:
        """
        mouse = {
            "nose": (mouse[0:2]),
            "left ear": (mouse[2:4]),
            "right ear": (mouse[4:6]),
            "neck": (mouse[6:8]),
            "left forepaw": (mouse[8:10]),
            "right forepaw": (mouse[10:12]),
            "center back": (mouse[12:14]),
            "left hindpaw": (mouse[14:16]),
            "right hindpaw": (mouse[16:18]),
            "tail base": (mouse[18:20]),
            "tail middle": (mouse[20:22]),
            "tail tip": (mouse[22:24]),
        }
        return mouse
