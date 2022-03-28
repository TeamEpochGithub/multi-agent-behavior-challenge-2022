import numpy as np

from lib.sequence import Sequence


def fix_zero_jump(seq: Sequence, verbose=False) -> None:
    """
    the given keypoints often suddenly jump to 0 for a few frames.
    this messes with velocities, for example

    method: when found a 0 keypoint, compare valid points with previous frame and approximate with
        new_point = prev_point + (avg displacement of valid points)
    works inplace
    :param seq:
    :param verbose:
    :return:
    """
    # first frame should be valid, so if it's not, take the earliest valid mouse
    for m in range(3):
        for f in range(0, seq.frames.shape[0]):
            if not np.any(seq.get_mouse(m, f) == 0):
                seq.set_mouse(m, 0, seq.get_mouse(m, f))
                break
            if f == seq.frames.shape[0] - 1:
                if verbose:
                    print("encountered a zero in all mouse positions, keeping original")
                return

    for f in range(1, seq.frames.shape[0]):
        for m in range(3):
            current_pos = seq.get_mouse(m, f)
            if np.any(current_pos == 0):
                prev_pos = seq.get_mouse(m, f - 1)
                # compute avg displacement of valid points
                displacements = []
                for body_part in range(12):
                    if current_pos[body_part * 2] != 0 and current_pos[body_part * 2 + 1] != 0:
                        displacements.append(
                            [
                                current_pos[body_part * 2] - prev_pos[body_part * 2],
                                current_pos[body_part * 2 + 1] - prev_pos[body_part * 2 + 1],
                            ]
                        )
                if len(displacements) == 0:
                    avg_disp = np.array([0, 0])
                else:
                    avg_disp = np.mean(displacements, 0)

                # fix broken points
                for body_part in range(12):
                    if current_pos[body_part * 2] == 0 or current_pos[body_part * 2 + 1] == 0:
                        current_pos[body_part * 2] = prev_pos[body_part * 2] + avg_disp[0]
                        current_pos[body_part * 2 + 1] = prev_pos[body_part * 2 + 1] + avg_disp[1]

                seq.set_mouse(m, f, current_pos)
