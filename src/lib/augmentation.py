import torch 
import numpy as np
import pickle
from scipy.ndimage import gaussian_filter1d

# https://math.stackexchange.com/questions/1746465/the-expression-for-reflection-of-a-ray-line-axbyc-0-reflected-by-a-mirror-wh
# https://over.wiki/ask/how-to-make-a-move-relative-to-a-straight-line-ax-by-c-0-affine-transformations/
# https://arxiv.org/pdf/2011.13917.pdf

def reflect_points(keypoints, a, b, c):
    """
    Reflects the keypoints with respect to the line. Normalize the equation by dividing it by sqrt(a^2 + b^2)
    a * x + b * y + c = 0

    Horizontal reflection: a = 0, b = 1, c = -FRAME_HEIGHT // 2
    Vertical reflection: a = 1, b = 0, c = -FRAME_WIDTH // 2

    :param keypoints: keypoints of all 3 mice in a single frame
    :param a: look at the equation in the description
    :param b: look at the equation in the description
    :param c: look at the equation in the description
    :return: the new set of keypoints
    """
    new_keypoints =  np.zeros(keypoints.shape)

    # normalize the equation
    m = np.sqrt(a * a + b * b)
    a = a / m
    b = b / m
    c = c / m

    d = a * keypoints[:, :, :, 0] + b * keypoints[:, :, :, 1] + c

    new_keypoints[:, :, :, 0] = keypoints[:, :, :, 0] - 2 * a * d
    new_keypoints[:, :, :, 1] = keypoints[:, :, :, 1] - 2 * b * d

    return new_keypoints





