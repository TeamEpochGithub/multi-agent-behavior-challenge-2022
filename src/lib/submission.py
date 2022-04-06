import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from lib.perceiver_data import dataset_from_frames


def find_seq(name, sub_sequences):
    """
    Finds the index of the sequence.
    Useful for features computed outside the function submission_embeddings
    :param name: id of the sequence
    :param sub_sequences: list of sequences
    """
    for i in range(len(sub_sequences)):
        if sub_sequences[i].name == name:
            return i

    raise ValueError


def submission_embeddings(
    config: dict, sub_clips: dict, model: nn.Module, sub_seq: list, func=None,
    embd: list = None
) -> dict:
    """
    Creates a ready for submission dict with an arbitrary model.
    (works only for the perceiver for now)
    :param config: configuration dict for different variables
    :param sub_clips: submission_clips
    :param model: trained model for submission
    :param sub_seq: list of sequences, used for find_seq function
    :param func: list of single-frame feature functions
    :param embd: list of np.ndarrays of precomputed features (e.g velocity)
    :return: dict ready for submission
    (however should be validated with the validate function before using it)
    """

    embeddings_model_size = config["model_embd_size"]
    features_size = config["feature_size"]
    sub_clips_items = sub_clips["sequences"].items()
    num_total_frames = np.sum([seq["keypoints"].shape[0] for _, seq in sub_clips_items])
    embeddings_size = embeddings_model_size + features_size
    embeddings_array = np.empty((num_total_frames, embeddings_size), dtype=np.float32)

    if embeddings_size > 128:
        raise ValueError(f"The maximum number of embeddings is 128, you have {embeddings_size}")

    frame_number_map = {}
    start = 0
    seq_len = config["seq_len"]

    for sequence_key, sequence in tqdm(zip(sub_clips["sequences"], sub_seq)):

        frames_seq, _ = dataset_from_frames(sequence.frames, config["length"],
                                            config["every Nth"], config["stride"], 0)
        frames_seq = torch.tensor(frames_seq, dtype=torch.float32).cuda()
        keypoints = sub_clips["sequences"][sequence_key]["keypoints"]
        embeddings = np.empty((len(keypoints), embeddings_size), dtype=np.float32)

        embs = model(
            frames_seq, return_embeddings=config["return_embeddings"]
        )

        embs = embs.detach().cpu().numpy()
        embeddings[:, :embeddings_model_size] = np.repeat(embs, config["stride"], axis=0)

        seq_index = find_seq(sequence_key, sub_seq)
        # in the notebook you have full energies here, needs to be passed in embd
        # also reshaping needs to be done before and passed directly in embd
        last = embeddings_model_size
        # precomputed features 
        for e in embd:
            if e[seq_index].shape != (3,):
                reshaped_e = e[seq_index].reshape(1800, -1)
                embeddings[:, last : last + reshaped_e.shape[1]] = reshaped_e
                last += reshaped_e.shape[1]
            else:
                embeddings[:, last:last + 3] = e[seq_index]
                last += 3

        # single frame features
        if config["use_single_frame"]:
            if func == None:
                raise ValueError("You have not passed any functions in func.")

            for i in range(len(keypoints)):
                lastx = last
                for f in func:
                    temp = f(keypoints[i].flatten())
                    embeddings[i, lastx: lastx + temp.shape[0]] = temp
                    lastx = lastx + temp.shape[0]

        end = start + len(keypoints)
        embeddings_array[start:end] = embeddings
        frame_number_map[sequence_key] = (start, end)
        start = end

    assert end == num_total_frames
    submission_dict = {"frame_number_map": frame_number_map, "embeddings": embeddings_array}

    if validate_submission(submission_dict, sub_clips) == False:
        raise Exception("Your submission dictionary did not pass the validation script")

    return submission_dict


def validate_submission(submission, submission_clips):
    """
    Checks that the submission dict has all the specific reqs for a submission.
    """
    if not isinstance(submission, dict):
        print("Submission should be dict")
        return False

    if "frame_number_map" not in submission:
        print("Frame number map missing")
        return False

    if "embeddings" not in submission:
        print("Embeddings array missing")
        return False
    elif not isinstance(submission["embeddings"], np.ndarray):
        print("Embeddings should be a numpy array")
        return False
    elif not len(submission["embeddings"].shape) == 2:
        print("Embeddings should be 2D array")
        return False
    elif not submission["embeddings"].shape[1] <= 128:
        print("Embeddings too large, max allowed is 128")
        return False
    elif not isinstance(submission["embeddings"][0, 0], np.float32):
        print("Embeddings are not float32")
        return False

    total_clip_length = 0
    for key in submission_clips["sequences"]:
        start, end = submission["frame_number_map"][key]
        clip_length = submission_clips["sequences"][key]["keypoints"].shape[0]
        total_clip_length += clip_length
        if not end - start == clip_length:
            print(f"Frame number map for clip {key} does not match clip length")
            return False

    if not len(submission["embeddings"]) == total_clip_length:
        print("Emebddings length doesn't match submission clips total length")
        return False

    if not np.isfinite(submission["embeddings"]).all():
        print("Emebddings contains NaN or infinity")
        return False

    print("All checks passed")
    return True


def save_submission_file(filename, sub_dict: dict):
    """
    Saves the submission dict as an npy file.
    :param filename: name of the new submission file
    :param sub_dict: dict for submission
    :return:
    """
    np.save(filename, sub_dict)
