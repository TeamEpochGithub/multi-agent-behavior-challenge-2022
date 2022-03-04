"""
Module docstring
"""
from lib.utils import load_data, visualize_movements


def main():
    submission_clips, user_train = load_data()

    print("Dataset keys - ", submission_clips.keys())
    print("Number of submission sequences - ", len(submission_clips['sequences']))

    sequence_names = list(submission_clips["sequences"].keys())
    sequence_key = sequence_names[0]
    single_sequence = submission_clips["sequences"][sequence_key]["keypoints"]
    print("Sequence name - ", sequence_key)
    print("Single Sequence shape ", single_sequence.shape)
    print(f"Number of Frames in {sequence_key} - ", len(single_sequence))

    class_to_number = {s: i for i, s in enumerate(user_train['vocabulary'])}
    number_to_class = {i: s for i, s in enumerate(user_train['vocabulary'])}

    print(f"Labels given and their id:{class_to_number}")

    # Call this function in the notebook for animation
    visualize_movements(user_train, number_to_class)


if __name__ == "__main__":
    print("Running Main")
    main()
