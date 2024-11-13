import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

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

# RNN Model for classification
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  
        self.rnn = nn.GRU(embedding_dim, hidden_size, batch_first=True)  
        self.fc = nn.Linear(hidden_size, 2)  

    def forward(self, x):
        embedded = self.embedding(x)  
        _, hidden = self.rnn(embedded)  
        output = self.fc(hidden[-1])  
        return output

# Training loop
def train_rnn_classifier(train_data, train_labels, model, criterion, optimizer):
    model.train()
    for epoch in range(NUM_EPOCHS):
        for i in range(0, len(train_data), BATCH_SIZE):
            batch_data = train_data[i:i+BATCH_SIZE]
            batch_labels = train_labels[i:i+BATCH_SIZE]

            # Zero gradients, forward pass, backward pass, optimize
            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item()}')

# Model evaluation function
def evaluate_model(test_data, test_labels, model):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i in range(0, len(test_data), BATCH_SIZE):
            batch_data = test_data[i:i+BATCH_SIZE]
            batch_labels = test_labels[i:i+BATCH_SIZE]
            output = model(batch_data)
            predicted = torch.argmax(output, dim=1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        accuracy = correct / total
        print(f'Accuracy: {accuracy * 100:.2f}%')
        return accuracy


# Loading the data for training and testing
train_data, train_labels = preprocess_data('./data/train-vowel-examples.txt')
test_data, test_labels = preprocess_data('./data/dev-vowel-examples.txt')

# Initialize the model, loss function, and optimizer
model = RNNClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE)
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training the model
train_rnn_classifier(train_data, train_labels, model, criterion, optimizer)

# evaluate the model
evaluate_model(test_data, test_labels, model)
