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


def apply_single_frame(seq: Sequence, func: Callable) -> np.ndarray:
    """
    Runs a single-frame feature on all frames of the given sequence
    :param seq:
    :param func:
    :return: np array of size (amount_of_frames x return_shape_of_func)
    """
    results = []
    for i in range(len(seq.frames)):
        results.append(func(seq.frames[i]))
    return np.array(results)
