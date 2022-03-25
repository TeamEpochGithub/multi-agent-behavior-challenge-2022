import numpy as np


def dataset_from_frames(
    frames: np.ndarray, length: int, every_nth: int, stride: int, pad_to_ans: int
):
    """
    Creates a dataset for future prediction models
    :param frames: np array of frames (potentially with other features) from 1 sequence
        not necessarily flat
    :param length: amount of frames in 1 X datapoint
    :param every_nth: include every n-th frame (1 - frames are consecutive)
    :param stride: gap between starts of different datapoints
    :param pad_to_ans: frames to skip between known frames and answer
    :return: dataset with X = (almost) consecutive frames
        y = a frame after the last frame of the corresponding X
    """
    if frames.shape[0] > 1800:
        print("Suspiciously many frames, are you sure they are all from 1 sequence?")

    total_length = (length - 1) * every_nth + 1 + pad_to_ans
    if total_length > frames.shape[0]:
        raise ValueError(f"total length of 1 X is too large: {total_length}")

    max_valid_start = frames.shape[0] - total_length
    dataset = np.empty((max_valid_start // stride + 1, length, *frames[0].shape))
    y = np.empty((max_valid_start // stride + 1, *frames[0].shape))

    i_datapoint = 0
    for start in range(0, max_valid_start, stride):
        end = start + (length - 1) * every_nth + 1
        dataset[i_datapoint] = frames[start:end:every_nth]
        y[i_datapoint] = frames[end + pad_to_ans]
        i_datapoint += 1
    return dataset, y
