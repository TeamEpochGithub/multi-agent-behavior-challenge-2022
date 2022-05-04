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
    neck2 = frames[:, 24 + 6: 24 + 8]
    neck3 = frames[:, 48 + 6: 48 + 8]
    necks = [neck1, neck2, neck3]
    nose1 = frames[:, 0:2]
    nose2 = frames[:, 24 + 0: 24 + 2]
    nose3 = frames[:, 48 + 0: 48 + 2]
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
            intermouse_norm = np.linalg.norm(intermouse, axis=1)
            result[:, m1, m2, 0] = intermouse_norm

            # take diagonal values of matrix multiplication to get correct dot products
            dot_product = np.dot(direction, intermouse.T).diagonal()
            # this may give division by zero warnings
            dot_norm = dot_product / direction_norm / intermouse_norm
            # replace NaNs with zeros
            dot_norm = np.nan_to_num(dot_norm)
            # in case floating point computation takes over 1
            dot_norm = np.clip(dot_norm, -1, 1)
            angle = np.arccos(dot_norm)
            result[:, m1, m2, 1] = angle
    return result


def mouse_length(seq: Sequence) -> np.ndarray(dtype=float, shape=(1800, 3)):
    """
    Calculate mouse length for each mouse for each frame
    :param_seq: sequence
    :return:
    """
    frames = seq.frames
    nose1 = frames[:, 0:2]
    nose2 = frames[:, 24 + 0: 24 + 2]
    nose3 = frames[:, 48 + 0: 48 + 2]
    noses = [nose1, nose2, nose3]
    tail1 = frames[:, 18:20]
    tail2 = frames[:, 24 + 18: 24 + 20]
    tail3 = frames[:, 48 + 18: 48 + 20]
    tails = [tail1, tail2, tail3]
    result = np.zeros((1800, 3), dtype=float)
    for m in range(3):
        result[:, m] = np.linalg.norm(noses[m] - tails[m])
    return result


def mean_reach(seq: Sequence) -> np.ndarray(dtype=float, shape=(1800, 3)):
    """
    Calculate how spread out the mouse is (mean distance of points to centre)
    :param_seq: sequence
    :return: float array
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
    """
    frames = seq.frames

    nose1 = frames[:, 0:2]
    nose2 = frames[:, 24 + 0: 24 + 2]
    nose3 = frames[:, 48 + 0: 48 + 2]
    noses = [nose1, nose2, nose3]

    lforepaw1 = frames[:, 8:10]
    lforepaw2 = frames[:, 24 + 8: 24 + 10]
    lforepaw3 = frames[:, 48 + 8: 48 + 10]
    lforepaws = [lforepaw1, lforepaw2, lforepaw3]

    rforepaw1 = frames[:, 10:12]
    rforepaw2 = frames[:, 24 + 10: 24 + 12]
    rforepaw3 = frames[:, 48 + 10: 48 + 12]
    rforepaws = [rforepaw1, rforepaw2, rforepaw3]

    lhindpaw1 = frames[:, 14:16]
    lhindpaw2 = frames[:, 24 + 14: 24 + 16]
    lhindpaw3 = frames[:, 48 + 14: 48 + 16]
    lhindpaws = [lhindpaw1, lhindpaw2, lhindpaw3]

    rhindpaw1 = frames[:, 16:18]
    rhindpaw2 = frames[:, 24 + 16: 24 + 18]
    rhindpaw3 = frames[:, 48 + 16: 48 + 18]
    rhindpaws = [rhindpaw1, rhindpaw2, rhindpaw3]

    tail1 = frames[:, 18:20]
    tail2 = frames[:, 24 + 18: 24 + 20]
    tail3 = frames[:, 48 + 18: 48 + 20]
    tails = [tail1, tail2, tail3]

    keypoints = [noses, lforepaws, rforepaws, lhindpaws, rhindpaws, tails]

    center1 = frames[:, 12:14]
    center2 = frames[:, 24 + 12: 24 + 14]
    center3 = frames[:, 48 + 12: 48 + 14]
    centers = [center1, center2, center3]

    result = np.zeros((1800, 3), dtype=float)

    for m in range(3):
        distances = []
        for keypoint in keypoints:
            distances.append(np.linalg.norm(centers[m] - keypoint[m], axis=1))
        result[:, m] = np.mean(distances, axis=0)
    return result


def head_angle(seq: Sequence) -> np.ndarray(dtype=float, shape=(1800, 3)):
    frames = seq.frames

    nose1 = frames[:, 0:2]
    nose2 = frames[:, 24 + 0: 24 + 2]
    nose3 = frames[:, 48 + 0: 48 + 2]
    noses = [nose1, nose2, nose3]

    neck1 = frames[:, 6:8]
    neck2 = frames[:, 24 + 6: 24 + 8]
    neck3 = frames[:, 48 + 6: 48 + 8]
    necks = [neck1, neck2, neck3]

    tail1 = frames[:, 18:20]
    tail2 = frames[:, 24 + 18: 24 + 20]
    tail3 = frames[:, 48 + 18: 48 + 20]
    tails = [tail1, tail2, tail3]

    result = np.zeros((1800, 3), dtype=float)
    for m in range(3):
        body_dirs = necks[m] - tails[m]
        head_dirs = noses[m] - necks[m]

        body_dir_norm = np.linalg.norm(body_dirs, axis=1)
        head_dir_norm = np.linalg.norm(head_dirs, axis=1)

        # take diagonal values of matrix multiplication to get correct dot products
        dot_product = np.dot(body_dirs, head_dirs.T).diagonal()
        # this may give division by zero warnings
        dot_norm = dot_product / body_dir_norm / head_dir_norm
        # replace NaNs with zeros
        dot_norm = np.nan_to_num(dot_norm)
        # in case floating point computation takes over 1
        dot_norm = np.clip(dot_norm, -1, 1)
        angle = np.arccos(dot_norm)
        result[:, m] = angle
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
