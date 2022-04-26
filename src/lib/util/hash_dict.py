from collections.abc import Iterable


def hash_dict(d):
    """
    Produces a hash of a dictionary that can have lists as values
    :param d: the dict
    :return: absolute value of the hash
    """
    return abs(
        hash(
            tuple(
                map(
                    lambda x: (x[0], tuple(x[1]) if isinstance(x[1], Iterable) else x[1]), d.items()
                )
            )
        )
    )
