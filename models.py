import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import Indexer

class FrequencyBasedClassifier:
    def __init__(self, majority_class):
        self.majority_class = majority_class

    def predict(self, inputs):
        return [self.majority_class] * len(inputs)

def train_frequency_based_classifier(train_cons_exs, train_vowel_exs):
    # Determine the majority class based on the training data
    if len(train_vowel_exs) >= len(train_cons_exs):
        majority_class = 1  # Vowel class
    else:
        majority_class = 0  # Consonant class

    # Initialize the classifier with the majority class
    return FrequencyBasedClassifier(majority_class)

class RNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)  # LSTM layer
        self.fc = nn.Linear(hidden_dim, output_dim)  # Fully connected layer

    def forward(self, x):
        _, (hidden, _) = self.rnn(x)  # LSTM outputs hidden state and cell state
        out = self.fc(hidden[-1])
        return out

def train_rnn_classifier(num_epochs, train_exs_indices, train_labels, vocab_index):
    # Hyperparameters
    input_dim = len(vocab_index)  # Input dimension size
    hidden_dim = 100  # Increased hidden layer dimension for complexity
    output_dim = 2  # Output dimension (binary classification: vowel or consonant)

    # Initialize model, loss, and optimizer
    model = RNNClassifier(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Lower learning rate

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0

        for i, seq in enumerate(train_exs_indices):
            # Convert sequence to one-hot encoded tensors
            inputs = torch.eye(input_dim)[seq].unsqueeze(0)  # One-hot encoding
            labels = torch.tensor([train_labels[i]], dtype=torch.long)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accuracy calculation
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

        # Print training progress
        epoch_loss = total_loss / len(train_exs_indices)
        accuracy = correct / len(train_exs_indices)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')

    return model

