import numpy as np
import cv2 as cv
import copy

from lib.sequence import Sequence


def calc_velocities(seq: Sequence):
    """
    calculates velocity vectors (displacement of neck per frame) for all 3 mice
    velocity for the 0th frame is assumed the same as for frame 1 (probably better than just zeros)
    :param seq: 1 sequence
    :return: np array of shape (1800, 3, 2)
    """
    velocities = np.empty((1800, 3, 2))
    for i in range(1, 1800):
        for m in range(3):
            velocities[i][m] = Sequence.name_mouse(seq.get_mouse(m, i))['neck'] \
                               - Sequence.name_mouse(seq.get_mouse(m, i-1))['neck']
        if np.any(np.abs(velocities[i]) > 25):
            print(f'outlier at {i}: \n{velocities[i]}')
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
    return np.mean(np.apply_along_axis(lambda v: np.sqrt(v[0]**2 + v[1]**2), 2, velocities), 0)


def egocentric_align(frame, base_animal=0, new_origin=False, origin=None):
    """

    This function aims to transform the initial frame system of axis from allocentric to egocentric coordinates. 
    As such, the new axis will be in the direction of the vector defined by the nose keypoint in the frame and new origin. (new_origin = True) 
    If the origin remains the same (new_origin = False), nose and neck keypoints are used. 
    The other axis needs to be orthogonal to the first one and pass through the origin.     

    Here, https://github.com/LINCellularNeuroscience/VAME/ the rotation and translation with respect to some origin O(x0,y0) is solved using
    cv2.getRotationMatrix2D(). The function requires an angle theta for rotation. 

    :param frame: a single frame from a sequence. expected dimensions (3, 12, 2)
    :param new_origin: changes the origin of the plane to the keypoint with index origin_keypoint
    :param origin: coordinates of the keypoint to be used as origin for the new egocentric axis alignment. array expected
    :param base_animal: index of the animal by which the axis are aligned. expected range [0,3)
    :return: returns 
    """

    if base_animal > 2:
        raise ValueError("The paramater should be in the range [0,3)")

    if frame.shape != (3,12,2):
        raise ValueError("The frame should have shape: (3, 12, 2)")

    #if new_origin == True and origin == None:
    #    raise ValueError("origin_keypoint needs to be specified if new_origin=True")


    nose = frame[base_animal][0]

    if new_origin == True:
        slopey = (nose[1] - origin[1]) / (nose[0] - origin[0]) # slope (y-axis)
        slopex = -1 / slopey # slope of the perpendicular (x-axis)
        theta = np.arctan(slopex) # angle of rotation
        print(f"Rotation angle in radians: {theta}")
        
        # before using the rotation matrix theta needs to be converted into degrees
        theta = np.rad2deg(theta)
        
        print(f"Rotation angle in degrees: {theta}") 
        rotM = cv.getRotationMatrix2D(center = (int(origin[0]), int(origin[1])) , angle = theta, scale = 1) # rotation and translation matrix
    else:
        neck = frame[base_animal][3]
        slopey = (nose[1] - neck[1]) / (nose[0] - neck[0]) # slope (y-axis)
        slopex = -1 / slopey # slope of the perpendicular line (x-axis)
        theta = np.arctan(slopex) # angle of rotation
        print(f"Rotation angle in radians: {theta}")
        # before using the rotation matrix theta needs to be converted into degrees
        theta = np.rad2deg(theta)
        print(f"Rotation angle in degrees: {theta}") 
        rotM = cv.getRotationMatrix2D(center = (850,1), angle = theta, scale = 1) # rotation and translation matrix


    new_frame = copy.deepcopy(frame)
    new_frame = new_frame.astype(np.float64)

    for animal_id in range(0,3):
        for keypt in range(0,12):
            coord = np.concatenate((new_frame[animal_id][keypt], np.ones(1))) # add one dimension (for translation)
            coord = coord.reshape(-1, 1) # reshaping data for multiplication 

            new_frame[animal_id][keypt] = list(rotM.dot(coord)) # matrix-vector multiplication 

    
    return new_frame