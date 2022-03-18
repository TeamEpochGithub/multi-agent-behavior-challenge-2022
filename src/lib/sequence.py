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
        return self.frames[frame_idx][mouse_idx * 24 : (mouse_idx + 1) * 24]

    def set_mouse(self, mouse_idx, frame_idx, mouse: np.ndarray):
        if mouse.shape != (24,):
            raise ValueError(f"Incorrect mouse shape, expected (24, ), got {mouse.shape}")
        self.frames[frame_idx][mouse_idx * 24 : (mouse_idx + 1) * 24] = mouse

    def convert_to_vame(self, single_mouse_embedding):
        # Determines if the mice are input into the model one by one
        num_frames = 1800
        num_mice = 3
        # Write data header
        data_arr = [np.repeat(["x", "y", "likelihood"], 12)]
        for mouse in range(num_mice):
            for f_num in range(num_frames):
                mouse_kpts = self.get_mouse(0, f_num)
                likelihood_vals = list(range(2, 24, 2))
                row = np.insert(mouse_kpts, likelihood_vals, 1.0)
                data_arr.append(row)
        data_np_arr = np.array(data_arr)
        # df = pd.DataFrame(data_np_arr, columns=[''])
        np.savetxt("file.csv", data_np_arr, delimiter=",")
        return True

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
