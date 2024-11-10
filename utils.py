import numpy as np
from collections import Counter

class Indexer(object):
    """
    Bijection between objects and integers starting at 0. Useful for mapping
    labels, features, etc. into coordinates of a vector space.
    """
    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def __repr__(self):
        return str([str(self.get_object(i)) for i in range(0, len(self))])

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.objs_to_ints)

    def get_object(self, index):
        """
        :param index: integer index to look up
        :return: Returns the object corresponding to the particular index or None if not found
        """
        if index not in self.ints_to_objs:
            return None
        else:
            return self.ints_to_objs[index]

    def contains(self, object):
        """
        :param object: object to look up
        :return: Returns True if it is in the Indexer, False otherwise
        """
        return self.index_of(object) != -1

    def index_of(self, object):
        """
        :param object: object to look up
        :return: Returns -1 if the object isn't present, index otherwise
        """
        if object not in self.objs_to_ints:
            return -1
        else:
            return self.objs_to_ints[object]

    def add_and_get_index(self, object, add=True):
        """
        Adds the object to the index if it isn't present, always returns a nonnegative index
        :param object: object to look up or add
        :param add: True by default, False if we shouldn't add the object. If False, equivalent to index_of.
        :return: The index of the object
        """
        if not add:
            return self.index_of(object)
        if object not in self.objs_to_ints:
            new_idx = len(self.objs_to_ints)
            self.objs_to_ints[object] = new_idx
            self.ints_to_objs[new_idx] = object
        return self.objs_to_ints[object]


def get_frequency_features(examples):
    """
    Convert a list of text examples into frequency-based features.
    :param examples: List of text examples (strings)
    :return: NumPy array of feature vectors where each feature is the frequency of a character
    """
    # Define the list of characters that will be used in the feature vector
    all_chars = 'abcdefghijklmnopqrstuvwxyz '  # characters we care about (lowercase + space)
    char_to_index = {char: i for i, char in enumerate(all_chars)}

    features = []
    for example in examples:
        # Initialize a feature vector for each example
        feature_vector = np.zeros(len(char_to_index))

        # Count the frequency of each character in the example
        char_count = Counter(example)

        # Populate the feature vector with character counts
        for char, count in char_count.items():
            if char in char_to_index:
                feature_vector[char_to_index[char]] = count

        features.append(feature_vector)

    return np.array(features)
