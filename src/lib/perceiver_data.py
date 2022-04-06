from typing import List

import numpy as np
import torch
from tqdm import tqdm
from sklearn.preprocessing import minmax_scale

from lib.sequence import Sequence


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


def _gen_tensor_dataset(sequences, dataset_config):
    x = []
    y = []
    for i in tqdm(range(len(sequences))):
        curr_x, curr_y = dataset_from_frames(
            sequences[i].frames,
            dataset_config["length"],
            dataset_config["every Nth"],
            dataset_config["stride"],
            dataset_config["pad to ans"],
        )
        x.append(curr_x)
        y.append(curr_y)
    x_shape = (len(x) * x[0].shape[0], dataset_config["length"], -1)
    y_shape = (len(x) * x[0].shape[0], -1)
    x = torch.tensor(np.array(x).reshape(x_shape), dtype=torch.float32)
    y = torch.tensor(np.array(y).reshape(y_shape), dtype=torch.float32)
    return x, y


def full_tensor_train_test(
        sequences: List[Sequence], dataset_config: dict, test_fraction=0.2, seed=42
) -> [torch.tensor] * 4:
    """
    Creates a dataset from multiple sequences and does a ttsplit, ensuring the frames from one
    sequence are either only in train or in test
    :param sequences: list of sequences
    :param dataset_config: dict with fields: length, every Nth, stride, pad to ans
    :param test_fraction: part of data for test set
    :param seed: random seed
    :return: 4 tensors
    """
    sequences = sequences.copy()
    np.random.seed(seed)
    np.random.shuffle(sequences)
    num_test = round(test_fraction * len(sequences))
    test = sequences[:num_test]
    train = sequences[num_test:]

    x_train, y_train = _gen_tensor_dataset(train, dataset_config)
    x_test, y_test = _gen_tensor_dataset(test, dataset_config)

    x_train_rs = torch.reshape(x_train, (x_train.shape[0] * x_train.shape[1], x_train.shape[2]))
    x_test_rs = torch.reshape(x_test, (x_test.shape[0] * x_test.shape[1], x_test.shape[2]))

    x_all = torch.cat((x_train_rs, x_test_rs, y_train, y_test), dim=0)
    x_all_scaled = minmax_scale(x_all)
    x_all_scaled = torch.tensor(x_all_scaled)
    x_train_rs_sc, x_test_rs_sc, y_train, y_test = torch.split(x_all_scaled, [x_train_rs.shape[0],
                                                                              x_test_rs.shape[0],
                                                                              y_train.shape[0],
                                                                              y_test.shape[0]])

    x_train = torch.reshape(x_train_rs_sc, x_train.shape)
    x_test = torch.reshape(x_test_rs_sc, x_test.shape)

    return x_train, x_test, y_train, y_test
