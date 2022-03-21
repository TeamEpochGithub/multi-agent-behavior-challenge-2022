from typing import Callable

import numpy as np

from lib.sequence import Sequence


def calc_velocities(seq: Sequence, verbose=False):
    """
    calculates velocity vectors (displacement of neck per frame) for all 3 mice
    velocity for the 0th frame is assumed the same as for frame 1 (probably better than just zeros)
    :param seq: 1 sequence
    :param verbose:
    :return: np array of shape (1800, 3, 2)
    """
    velocities = np.empty((1800, 3, 2))
    for i in range(1, 1800):
        for m in range(3):
            velocities[i][m] = (
                Sequence.name_mouse(seq.get_mouse(m, i))["neck"]
                - Sequence.name_mouse(seq.get_mouse(m, i - 1))["neck"]
            )
        if np.any(np.abs(velocities[i]) > 25) and verbose:
            print(f"outlier at {i}: \n{velocities[i]}")
            print(Sequence.name_mouse(seq.get_mouse(0, i))["neck"])
            print(Sequence.name_mouse(seq.get_mouse(1, i))["neck"])
            print(Sequence.name_mouse(seq.get_mouse(2, i))["neck"])
    velocities[0] = velocities[1]
    return velocities


def calc_energy(velocities):
    """
    idea: if a mouse is running during the whole video,
        it might be different from when it made a short sprint
    :param velocities: output of calc_velocities function
    :return: average speed over sequence per mouse
    """
    return np.mean(np.apply_along_axis(lambda v: np.sqrt(v[0] ** 2 + v[1] ** 2), 2, velocities), 0)


def vectorized_mice_distance_angle(seq: Sequence) -> np.ndarray(dtype=float, shape=(1800, 3, 3, 2)):
    """
    Hopefully faster implementation of mice_distance_angle
    :param seq: sequence
    :return:
    """
    frames = seq.frames
    # get necessary keypoints without dictionaries
    neck1 = frames[:, 6:8]
    neck2 = frames[:, 24 + 6 : 24 + 8]
    neck3 = frames[:, 48 + 6 : 48 + 8]
    necks = [neck1, neck2, neck3]
    nose1 = frames[:, 0:2]
    nose2 = frames[:, 24 + 0 : 24 + 2]
    nose3 = frames[:, 48 + 0 : 48 + 2]
    noses = [nose1, nose2, nose3]
    if frames.shape[0] != 1800:
        raise ValueError
    result = np.zeros((1800, 3, 3, 2), dtype=float)
    for m1 in range(3):
        neck = necks[m1]
        nose = noses[m1]
        direction = nose - neck
        direction_norm = np.linalg.norm(direction, axis=1)
        for m2 in range(3):
            if m1 == m2:
                continue
            second_neck = necks[m2]
            intermouse = second_neck - neck
            result[:, m1, m2, 0] = np.linalg.norm(intermouse, axis=1)
            intermouse_norm = np.linalg.norm(intermouse, axis=1)
            # ake diagonal values of matrix multiplication to get correct dot products
            dot_product = np.dot(direction, intermouse.T).diagonal()
            # this may give division by zero warnings
            dot_norm = dot_product / direction_norm / intermouse_norm
            # fix NaNs to set zeros
            dot_norm = np.nan_to_num(dot_norm)
            # in case floating point computation takes over 1
            dot_norm = np.clip(dot_norm, -1, 1)
            angle = np.arccos(dot_norm)
            result[:, m1, m2, 1] = angle
    return result


def apply_single_frame(seq: Sequence, func: Callable) -> np.ndarray:
    """
    Runs a single-frame feature on all frames of the given sequence
    May be really slow
    :param seq:
    :param func:
    :return: np array of size (amount_of_frames x return_shape_of_func)
    """
    results = []
    for i in range(len(seq.frames)):
        results.append(func(seq.frames[i]))
    return np.array(results)
