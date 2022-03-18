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


def calc_mouse_length(frame: np.ndarray) -> np.ndarray(dtype=float, shape=(3)):
    """
    calculates the length of mice (length between root of tail and nose)
    :param frame: original keypoints - 72 numbers
    :return: float array
    """
    mice = [Sequence.mouse_from_frame(frame, 0),
            Sequence.mouse_from_frame(frame, 1),
            Sequence.mouse_from_frame(frame, 2)]
    lengths = np.zeros((3), dtype=float)
    for m in range(3):
        lengths[m] = np.linalg.norm(Sequence.name_mouse(mice[m])['nose'] - Sequence.name_mouse(mice[m])['tail base'])
    return lengths


def calc_mean_reach(frame: np.ndarray) -> np.ndarray(dtype=float, shape=(3)):
    """
    calculates the mean distance from keypoints to 'center back' for each mouse (excluding the tail)
    :param frame: original keypoints - 72 numbers
    :return: float array
    """
    mice = [Sequence.mouse_from_frame(frame, 0),
            Sequence.mouse_from_frame(frame, 1),
            Sequence.mouse_from_frame(frame, 2)]
    reach = np.zeros((3), dtype=float)
    for m in range(3):
        distances = []
        for keypoint in Sequence.name_mouse(mice[m]):
            if keypoint not in ['center back', 'tail middle', 'tail tip']:
                distances.append(np.linalg.norm(
                    Sequence.name_mouse(mice[m])[keypoint] - Sequence.name_mouse(mice[m])['center back']))
        reach[m] = np.mean(distances)

    return reach


def calc_head_angle(frame: np.ndarray) -> np.ndarray(dtype=float, shape=(3)):
    """
    calculates the angle of nose keypoint from the body axis
    angle does not differentiate between left and right
    angle is between 0 and pi
    :param frame: original keypoints - 72 numbers
    :return: float array
    """
    mice = [Sequence.mouse_from_frame(frame, 0),
            Sequence.mouse_from_frame(frame, 1),
            Sequence.mouse_from_frame(frame, 2)]
    result = np.zeros((3), dtype=float)
    for m in range(3):
        nose = Sequence.name_mouse(mice[m])['nose']
        neck = Sequence.name_mouse(mice[m])['neck']
        tail_base = Sequence.name_mouse(mice[m])['tail base']

        body_direction = neck - tail_base
        head_direction = nose - neck

        body_direction_norm = np.linalg.norm(body_direction)
        head_direction_norm = np.linalg.norm(head_direction)

        if body_direction_norm != 0 and head_direction_norm != 0:
            dot_product = np.dot(body_direction, head_direction) / \
                          body_direction_norm / head_direction_norm
            dot_product = np.clip(dot_product, -1, 1)
            angle = np.arccos(dot_product)
            result[m] = angle
        else:
            print("(tail_base -> neck) or (neck -> nose) vector has length 0")

    return result
