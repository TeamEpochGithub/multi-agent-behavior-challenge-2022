import numpy as np


class Sequence:
    def __init__(self, name, seq_array: np.ndarray, chasing_labels=None, light_label=None):
        self.name = name
        flat = seq_array.reshape((1800, 72))  # 72 = (num mice = 3) x (body parts = 12) x (x, y coordinate).
        self.original_keypoints = flat
        self.light_label = light_label
        self.chasing_labels = chasing_labels

    def get_mouse(self, idx, frame):
        """
        get first, second or third mouse
        :param idx: index of mouse
        :param frame: from which frame
        :return: np array with 24 numbers (nose_x, nose_y, left_ear_x, ...)
        """
        return self.original_keypoints[frame][idx * 24:(idx + 1) * 24]

    @staticmethod
    def name_mouse(mouse: np.ndarray) -> dict:
        mouse = {'nose': (mouse[0, 1]),
                 'left ear': (mouse[2, 3]), 'right ear': (mouse[4, 5]),
                 'neck': (mouse[6, 7]),
                 'left forepaw': (mouse[8, 9]), 'right forepaw': (mouse[10, 11]),
                 'center back': (mouse[12, 13]),
                 'left hindpaw': (mouse[14, 15]), 'right hindpaw': (mouse[16, 17]),
                 'tail base': (mouse[18, 19]), 'tail middle': (mouse[20, 21]), 'tail tip': (mouse[22, 23])}
        return mouse
