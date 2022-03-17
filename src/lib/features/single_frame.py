import numpy as np

from lib.sequence import Sequence


def mice_distance_angle(frame: np.ndarray) -> np.ndarray(dtype=float, shape=(3, 3, 2)):
    """
    Produces distances between necks of mice and angle of neck with respect to neck -> nose vector
    For index consistency, there are 3 values each mouse
     and a mouse has 0 distance and 0 angle to itself
    Angle (in radians) does not differentiate between left and right.
        Angle is between 0 and pi
    :param frame: original keypoints - 72 numbers
    :return: float array,
     [i][j][0] is distance between mouse i and j,
     [i][j][1] is angle between mouse i and j
    """
    mice = [Sequence.mouse_from_frame(frame, 0),
            Sequence.mouse_from_frame(frame, 1),
            Sequence.mouse_from_frame(frame, 2)]
    result = np.zeros((3, 3, 2), dtype=float)
    for m1 in range(3):
        neck = Sequence.name_mouse(mice[m1])['neck']
        nose = Sequence.name_mouse(mice[m1])['nose']
        direction = nose - neck
        direction_norm = np.linalg.norm(direction)
        for m2 in range(3):
            if m1 == m2:
                continue
            second_neck = Sequence.name_mouse(mice[m2])['neck']
            intermouse = second_neck - neck
            result[m1][m2][0] = np.linalg.norm(intermouse)
            intermouse_norm = np.linalg.norm(intermouse)
            if direction_norm != 0 and intermouse_norm != 0:
                dot_product = np.dot(direction, intermouse) / direction_norm / intermouse_norm
                # in case floating point computation takes over 1
                dot_product = np.clip(dot_product, -1, 1)
                angle = np.arccos(dot_product)
                result[m1][m2][1] = angle
            else:
                print("(neck -> nose) or (neck -> second neck) vector has length 0")
    return result
