import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Indexer:
    def __init__(self):
        self.obj_to_idx = {}
        self.idx_to_obj = []

    def add_and_get_index(self, obj):
        if obj not in self.obj_to_idx:
            self.obj_to_idx[obj] = len(self.idx_to_obj)
            self.idx_to_obj.append(obj)
        return self.obj_to_idx[obj]

    def get_index(self, obj):
        # Return the index of the object if it exists, otherwise -1 (or some default behavior)
        return self.obj_to_idx.get(obj, -1)

    def __len__(self):
        return len(self.idx_to_obj)


# Frequency-Based Model: A classifier based on frequency of characters
class FrequencyClassifier:
    def __init__(self):
        self.classifier = LogisticRegression(max_iter=1000)

    def fit(self, train_cons_exs, train_vowel_exs):
        cons_freq_features = get_frequency_features(train_cons_exs)
        vowel_freq_features = get_frequency_features(train_vowel_exs)
        X_train = np.concatenate((cons_freq_features, vowel_freq_features), axis=0)
        y_train = np.concatenate((np.zeros(len(cons_freq_features)), np.ones(len(vowel_freq_features))), axis=0)
        self.classifier.fit(X_train, y_train)

    def predict(self, text):
        features = get_frequency_features([text])
        return int(self.classifier.predict(features)[0])

# RNN-Based Model: A classifier using a Recurrent Neural Network (RNN) architecture

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        output = self.fc(rnn_out[:, -1, :])
        return output
def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    hidden_dim = 128
    embedding_dim = 64
    output_dim = 2
    batch_size = 54
    epochs = 50

    vocab_size = len(vocab_index)
    model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)

    cons_indices = [torch.tensor([vocab_index.add_and_get_index(char) for char in ex], dtype=torch.long) for ex in train_cons_exs]
    vowel_indices = [torch.tensor([vocab_index.add_and_get_index(char) for char in ex], dtype=torch.long) for ex in train_vowel_exs]
    all_data = cons_indices + vowel_indices
    all_labels = [0] * len(cons_indices) + [1] * len(vowel_indices)
    train_data = torch.utils.data.TensorDataset(torch.nn.utils.rnn.pad_sequence(all_data, batch_first=True),
                                                torch.tensor(all_labels, dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Training loop with loss and accuracy tracking
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for texts, labels in train_loader:
            texts = texts.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Evaluate the model on training and test sets after training
    model.eval()
    with torch.no_grad():
        # Training set evaluation
        train_preds, train_labels = [], []
        for texts, labels in train_loader:
            texts = texts.to(device)
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_accuracy = accuracy_score(train_labels, train_preds)
        print(f"Training Accuracy: {train_accuracy:.4f}")

        # Testing set evaluation
        dev_cons_indices = [torch.tensor([vocab_index.add_and_get_index(char) for char in ex], dtype=torch.long) for ex in dev_cons_exs]
        dev_vowel_indices = [torch.tensor([vocab_index.add_and_get_index(char) for char in ex], dtype=torch.long) for ex in dev_vowel_exs]
        dev_data = dev_cons_indices + dev_vowel_indices
        dev_labels = [0] * len(dev_cons_indices) + [1] * len(dev_vowel_indices)

        dev_data = torch.nn.utils.rnn.pad_sequence(dev_data, batch_first=True)
        dev_data = dev_data.to(device)
        dev_labels = torch.tensor(dev_labels, dtype=torch.long).to(device)

        outputs = model(dev_data)
        _, dev_preds = torch.max(outputs, 1)
        test_accuracy = accuracy_score(dev_labels.cpu(), dev_preds.cpu())
        print(f"Testing Accuracy: {test_accuracy:.4f}")

    return model


def train_frequency_based_classifier(train_cons_exs, train_vowel_exs):
    """
    Train the frequency-based classifier using the provided training data.

    :param train_cons_exs: List of training examples for consonants
    :param train_vowel_exs: List of training examples for vowels
    :return: Trained frequency-based classifier
    """
    model = FrequencyClassifier()  # Initialize the frequency classifier
    model.fit(train_cons_exs, train_vowel_exs)  # Train the classifier
    return model

#####################
# MODELS FOR PART 2 #
#####################

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse

# Task 2: Language Model (Base Class)
class LanguageModel(object):
    def get_log_prob_single(self, next_char, context):
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context):
        raise Exception("Only implemented in subclasses")

# Task 2: Uniform Language Model
class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_log_prob_single(self, next_char, context):
        return np.log(1.0 / self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0 / self.voc_size) * len(next_chars)

# Task 2: RNN Language Model (with Task 3 modifications)
class RNNLanguageModel(LanguageModel):
    def __init__(self, model_emb, model_dec, vocab_index, sos_token, eos_token):
        super(RNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab_index), model_emb)  # Embedding layer
        self.rnn = nn.RNN(model_emb, model_dec, batch_first=True)  # RNN layer
        self.fc = nn.Linear(model_dec, len(vocab_index))  # Fully connected layer to output vocabulary size

        self.vocab_index = vocab_index
        self.sos_token = sos_token
        self.eos_token = eos_token

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.rnn.hidden_size)

    def forward(self, input_seq, hidden_state):
        embedded = self.embedding(input_seq)
        output, hidden_state = self.rnn(embedded, hidden_state)
        output = self.fc(output)
        return output, hidden_state

    def get_log_prob_single(self, next_char, context):
        pass

    def get_log_prob_sequence(self, next_chars, context):
        pass

# Task 2: Parse Arguments for Command-Line Execution
def parse_args():
    parser = argparse.ArgumentParser(description='Language model arguments')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the training data')
    parser.add_argument('--dev_path', type=str, required=True, help='Path to the development data')
    parser.add_argument('--model', type=str, choices=['UNIFORM', 'RNN'], default='RNN', help='Model type')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model_emb', type=int, default=128, help='Embedding size')
    parser.add_argument('--model_dec', type=int, default=128, help='Hidden size of the RNN')
    return parser.parse_args()

# Task 2 & 3: Train the Language Model
# Task 2 & 3: Train the Language Model with Debug Prints
def train_lm(args, train_text, dev_text, vocab_index):
    chunk_size = 100
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.learning_rate

    # Add SOS and EOS tokens (Task 3)
    sos_token = vocab_index.add_and_get_index('<SOS>', add=True)
    eos_token = vocab_index.add_and_get_index('<EOS>', add=True)

    # Task 2: Initialize the RNN model
    if args.model == "RNN":
        model = RNNLanguageModel(args.model_emb, args.model_dec, vocab_index, sos_token, eos_token)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Convert train_text to indices based on vocab_index
    train_indices = [vocab_index.index_of(char) for char in train_text]

    # Function to calculate accuracy
    def calculate_accuracy(predictions, target):
        _, predicted = torch.max(predictions, dim=1)
        correct = (predicted == target).float()
        accuracy = correct.sum() / len(correct)
        return accuracy.item()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        hidden_state = None  # Initialize hidden state
        total_loss = 0
        total_accuracy = 0

        # Chunk the data and loop over each chunk
        for i in range(0, len(train_indices) - chunk_size, chunk_size):
            chunk = train_indices[i:i + chunk_size]
            input_seq = [sos_token] + chunk[:-1]  # Add <SOS> token at the start of the input
            target_seq = chunk[1:] + [eos_token]  # Add <EOS> token at the end of the target

            input_seq = torch.tensor(input_seq).unsqueeze(0)  # Add batch dimension
            target_seq = torch.tensor(target_seq).unsqueeze(0)  # Add batch dimension

            if hidden_state is None:
                hidden_state = model.init_hidden(batch_size)
            else:
                hidden_state = hidden_state.detach()  # Detach hidden state to avoid backprop through entire history

            optimizer.zero_grad()

            # Forward pass through the model
            output, hidden_state = model(input_seq, hidden_state)
            output = output.view(-1, len(vocab_index))  # Flatten output for loss computation
            target_seq = target_seq.view(-1)  # Flatten target sequence for loss computation

            # Calculate loss and perform backpropagation
            loss = criterion(output, target_seq)
            loss.backward()
            optimizer.step()

            # Calculate accuracy for this batch
            accuracy = calculate_accuracy(output, target_seq)
            total_loss += loss.item()
            total_accuracy += accuracy

        avg_loss = total_loss / (len(train_indices) // chunk_size)
        avg_accuracy = total_accuracy / (len(train_indices) // chunk_size)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

        # After training, generate a sequence to test SOS/EOS handling
        generated_text = generate_sequence(model, vocab_index, max_length=100)
        print(f"Generated sequence: {generated_text}")

    return model

# Task 3: Generate Sequence using the Model
def generate_sequence(model, vocab_index, max_length=100):
    input_seq = [vocab_index.index_of('<SOS>')]
    generated_seq = []

    for _ in range(max_length):
        output, _ = model(torch.tensor(input_seq).unsqueeze(0))
        next_char_idx = torch.argmax(output, dim=2).item()

        next_char = vocab_index.char_of(next_char_idx)
        generated_seq.append(next_char)

        input_seq.append(next_char_idx)

        if next_char == '<EOS>':
            break

    return ''.join(generated_seq)

# Task 2 & 3: Indexer Class (for Vocab Index Management)
class Indexer:
    def __init__(self):
        self.index_to_char = []
        self.char_to_index = {}
        self.counter = 0

    def add_and_get_index(self, char, add=False):
        if char not in self.char_to_index:
            if add:
                self.char_to_index[char] = self.counter
                self.index_to_char.append(char)
                self.counter += 1
        return self.char_to_index[char]

    def index_of(self, char):
        return self.char_to_index.get(char, self.char_to_index.get('<UNK>'))

    def char_of(self, index):
        return self.index_to_char[index]

# Task 2 & 3: Main Function to Execute the Program
def main():
    args = parse_args()

    # Load the training and development data
    train_text = open(args.train_path).read()
    dev_text = open(args.dev_path).read()

    vocab_index = Indexer()
    vocab_index.add_and_get_index('<UNK>', add=True)  # Unknown token
    vocab_index.add_and_get_index('<SOS>', add=True)
    vocab_index.add_and_get_index('<EOS>', add=True)

    model = train_lm(args, train_text, dev_text, vocab_index)
    print("Model trained successfully.")

if __name__ == "__main__":
    main()


