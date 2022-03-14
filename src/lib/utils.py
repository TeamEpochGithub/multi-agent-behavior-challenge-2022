import copy
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation
from matplotlib import rc

from lib.sequence import Sequence

matplotlib.use("TkAgg")
rc('animation', html='jshtml')

# Note: Image processing may be slow if too many frames are animated.
# Note 2: This file contains the functionality found in the starter kit notebook
#  For further documentation on the steps, please check the notebook itself.
#  https://www.aicrowd.com/showcase/getting-started-mabe-2022-mouse-triplets-round-1

# Plotting constants
FRAME_WIDTH_TOP = 850
FRAME_HEIGHT_TOP = 850

M1_COLOR = 'lawngreen'
M2_COLOR = 'skyblue'
M3_COLOR = 'tomato'

class_to_color = {'other': 'white', 'attack': 'red', 'mount': 'green',
                  'investigation': 'orange'}

PLOT_MOUSE_START_END = [(0, 1), (1, 3), (3, 2), (2, 0),  # head
                        (3, 6), (6, 9),  # midline
                        (9, 10), (10, 11),  # tail
                        (4, 5), (5, 8), (8, 9), (9, 7), (7, 4)  # legs
                        ]

PATH_SUBMISSION_DATA = "../input/submission_data.npy"
PATH_USER_TRAIN = "../input/user_train.npy"


def load_data() -> (dict, dict):
    submission_clips = np.load(PATH_SUBMISSION_DATA, allow_pickle=True).item()
    user_train = np.load(PATH_USER_TRAIN, allow_pickle=True).item()
    return submission_clips, user_train


def visualize_movements(user_train, number_to_class):
    sequence_names = list(user_train['sequences'].keys())
    sequence_key = sequence_names[0]
    single_sequence = user_train["sequences"]['7MXIWNKUU6VTNGAUDICW']

    keypoint_sequence = single_sequence['keypoints']
    filled_sequence = fill_holes(keypoint_sequence)

    annotation_sequence = None  # single_sequence['annotations']

    ani = animate_pose_sequence(sequence_key,
                                filled_sequence,
                                start_frame=600,
                                stop_frame=700,
                                skip=1,
                                annotation_sequence=annotation_sequence,
                                class_to_color=class_to_color,
                                number_to_class=number_to_class)
    plt.show()
    # Display the animaion on colab
    return ani


def num_to_text(anno_list, number_to_class):
    return np.vectorize(number_to_class.get)(anno_list)


def fill_holes(data):
    clean_data = copy.deepcopy(data)
    for m in range(3):
        holes = np.where(clean_data[0, m, :, 0] == 0)
        if not holes:
            continue
        for h in holes[0]:
            sub = np.where(clean_data[:, m, h, 0] != 0)
            if (sub and sub[0].size > 0):
                clean_data[0, m, h, :] = clean_data[sub[0][0], m, h, :]
            else:
                return np.empty((0))

    for fr in range(1, np.shape(clean_data)[0]):
        for m in range(3):
            holes = np.where(clean_data[fr, m, :, 0] == 0)
            if not holes:
                continue
            for h in holes[0]:
                clean_data[fr, m, h, :] = clean_data[fr - 1, m, h, :]
    return clean_data


def set_figax():
    fig = plt.figure(figsize=(8, 8))

    img = np.zeros((FRAME_HEIGHT_TOP, FRAME_WIDTH_TOP, 3))

    ax = fig.add_subplot(111)
    ax.imshow(img)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    return fig, ax


def plot_mouse(ax, pose, color):
    # Draw each keypoint
    for j in range(10):
        ax.plot(pose[j, 0], pose[j, 1], 'o', color=color, markersize=3)

    # Draw a line for each point pair to form the shape of the mouse

    for pair in PLOT_MOUSE_START_END:
        line_to_plot = pose[pair, :]
        ax.plot(line_to_plot[:, 0], line_to_plot[
                                    :, 1], color=color, linewidth=1)


def animate_pose_sequence(video_name, seq, number_to_class, class_to_color, start_frame=0,
                          stop_frame=100, skip=0, annotation_sequence=None):
    # Returns the animation of the keypoint sequence between start frame
    # and stop frame. Optionally can display annotations.

    image_list = []

    counter = 0
    if skip:
        anim_range = range(start_frame, stop_frame, skip)
    else:
        anim_range = range(start_frame, stop_frame)

    for j in anim_range:
        if counter % 20 == 0:
            print("Processing frame ", j)
        fig, ax = set_figax()
        plot_mouse(ax, seq[j, 0, :, :], color=M1_COLOR)
        plot_mouse(ax, seq[j, 1, :, :], color=M2_COLOR)
        plot_mouse(ax, seq[j, 2, :, :], color=M3_COLOR)

        if annotation_sequence is not None:
            annot = annotation_sequence[j]
            annot = number_to_class[annot]
            plt.text(50, -20, annot, fontsize=16,
                     bbox=dict(facecolor=class_to_color[annot], alpha=0.5))

        ax.set_title(
            video_name + '\n frame {:03d}.png'.format(j))

        ax.axis('off')
        fig.tight_layout(pad=0)
        ax.margins(0)

        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(),
                                        dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(
            fig.canvas.get_width_height()[::-1] + (3,))

        image_list.append(image_from_plot)

        plt.close()
        counter = counter + 1

    # Plot animation.
    fig = plt.figure(figsize=(8, 8))
    plt.axis('off')
    im = plt.imshow(image_list[0])

    def animate(k):
        im.set_array(image_list[k])
        return im,

    ani = animation.FuncAnimation(fig, animate, frames=len(image_list), blit=True)
    return ani


def validate_submission(submission, submission_clips):
    if not isinstance(submission, dict):
        print("Submission should be dict")
        return False

    if 'frame_number_map' not in submission:
        print("Frame number map missing")
        return False

    if 'embeddings' not in submission:
        print('Embeddings array missing')
        return False
    elif not isinstance(submission['embeddings'], np.ndarray):
        print("Embeddings should be a numpy array")
        return False
    elif not len(submission['embeddings'].shape) == 2:
        print("Embeddings should be 2D array")
        return False
    elif not submission['embeddings'].shape[1] <= 128:
        print("Embeddings too large, max allowed is 128")
        return False
    elif not isinstance(submission['embeddings'][0, 0], np.float32):
        print("Embeddings are not float32")
        return False

    total_clip_length = 0
    for key in submission_clips['sequences']:
        start, end = submission['frame_number_map'][key]
        clip_length = submission_clips['sequences'][key]['keypoints'].shape[0]
        total_clip_length += clip_length
        if not end - start == clip_length:
            print(f"Frame number map for clip {key} doesn't match clip length")
            return False

    if not len(submission['embeddings']) == total_clip_length:
        print("Emebddings length doesn't match submission clips total length")
        return False

    if not np.isfinite(submission['embeddings']).all():
        print("Emebddings contains NaN or infinity")
        return False

    print("All checks passed")
    return True


def make_sequences():
    """
    2 lists of Sequence instances. train and test
    :return:
    """
    sub_clips, train_data = load_data()
    train_sequences = []
    for key, value in train_data['sequences'].items():
        labels = value['annotations']
        train_sequences.append(
            Sequence(key, value['keypoints'], labels[0], labels[1][0]))
    submission_sequences = []
    for key, value in sub_clips['sequences'].items():
        submission_sequences.append(
            Sequence(key, value['keypoints']))
    return train_sequences, submission_sequences
