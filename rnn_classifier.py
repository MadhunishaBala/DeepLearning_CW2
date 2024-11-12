import torch
import numpy as np

# constants
VOCAB_SIZE = 27  # Lowercase letters (a-z) + space
EMBEDDING_DIM = 16
HIDDEN_SIZE = 32
BATCH_SIZE = 64
NUM_EPOCHS = 10

# set of vowels and consonants
VOWELS = set("aeiou")
CONSONANTS = set("bcdfghjklmnpqrstvwxyz")

# a small function to map characters to indices
def char_to_idx(char):
    if char == ' ':
        return 0
    return ord(char) - ord('a') + 1 

# function for data pre-processing
def preprocess_data(filename):
    sequences = []
    labels = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()  # Remove any trailing spaces
            sequence = [char_to_idx(c) for c in line[:-1]]  # First 20 characters
            next_char = line[-1]  # Next character is the target
            sequences.append(sequence)
            label = 1 if next_char in VOWELS else 0  # 1 for vowels, 0 for consonants
            labels.append(label)
    return torch.tensor(sequences), torch.tensor(labels)
