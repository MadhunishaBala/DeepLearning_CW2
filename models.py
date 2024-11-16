#models.py

import numpy as np
import torch
import collections
import torch.nn as nn
import torch.optim as optim


#####################
# MODELS FOR PART 1 #
#####################

class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Only implemented in subclasses")


class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
    classify as consonant, otherwise as vowel.
    """
    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1

class RNNClassifier(ConsonantVowelClassifier, nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, vocab_index, num_layers, dropout, bidirectional):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Use GRU with dropout and optionally bidirectional
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout,
                          bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)  # Adjust for bidirectional GRU
        self.vocab_index = vocab_index  # Store vocab_index within the model

    def forward(self, context_tensor):
        embedded = self.embedding(context_tensor)  # Get embeddings for the input sequence
        _, hidden = self.gru(embedded)  # Output hidden state of GRU
        # If bidirectional, concatenate the last hidden states of both directions
        if isinstance(hidden, tuple):  # For GRU with num_layers > 1
            hidden = hidden[0]
        output = self.fc(hidden[-1])  # Feed the last hidden state to the FC layer
        return output  # Return logits (no softmax here)

    def predict(self, context):
        # Convert context string to indices
        context_indices = torch.tensor(
            [self.vocab_index.index_of(char) for char in context],
            dtype=torch.long
        ).unsqueeze(0)  # Add batch dimension
        output = self.forward(context_indices)
        prediction = torch.argmax(output, dim=1).item()  # Get the predicted class
        return prediction

def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)

def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    # Set default model parameters inside the function
    embedding_dim = 50
    hidden_dim = 50
    learning_rate = 0.001
    num_epochs = 10
    output_dim = 2  # Consonant or Vowel


    batch_size = 64
    num_layers = 2
    dropout = 0.5
    bidirectional = False

    # Get the vocab size
    vocab_size = len(vocab_index)

    # Initialize the model
    model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, vocab_index,
                          num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)

    # Prepare training data
    train_data = [(ex, 0) for ex in train_cons_exs] + [(ex, 1) for ex in train_vowel_exs]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Convert data to batches
    def get_batches(data, batch_size):
        # Shuffle data and create batches
        np.random.shuffle(data)
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            contexts, labels = zip(*batch)
            contexts_tensor = torch.tensor([[vocab_index.index_of(char) for char in context] for context in contexts], dtype=torch.long)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            yield contexts_tensor, labels_tensor

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for context_tensor, label_tensor in get_batches(train_data, batch_size):
            optimizer.zero_grad()
            output = model(context_tensor)
            loss = criterion(output, label_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            correct += (predicted == label_tensor).sum().item()
            total += label_tensor.size(0)

        # Calculate and print training accuracy
        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Validation after each epoch
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation for validation
            correct = 0
            total = 0
            for context, label in zip(dev_cons_exs + dev_vowel_exs, [0] * len(dev_cons_exs) + [1] * len(dev_vowel_exs)):
                context_indices = torch.tensor([vocab_index.index_of(char) for char in context], dtype=torch.long).unsqueeze(0)
                output = model(context_indices)  # No need to worry about GPU here

                _, predicted = torch.max(output, 1)

                # Convert label to a tensor before comparison
                label_tensor = torch.tensor([label], dtype=torch.long)

                correct += (predicted == label_tensor).sum().item()
                total += label_tensor.size(0)  # Now label_tensor is a tensor, so .size(0) works

            # Calculate and print validation accuracy
            val_accuracy = 100 * correct / total
            print(f"Validation Accuracy: {val_accuracy:.2f}%")

    return model

#####################
# MODELS FOR PART 2 #
#####################


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

#Nisha check
import numpy as np

def evaluate_perplexity_and_likelihood(model, test_data):
    log_prob_sum = 0
    token_count = 0

    # Iterate over each token in the test data
    for i in range(1, len(test_data)):
        # Calculate probability of current token given previous tokens
        prob = model.predict(test_data[:i], test_data[i])

        # Add log probability to the sum
        log_prob_sum += np.log(prob)
        token_count += 1

    # Calculate average log likelihood
    avg_log_likelihood = log_prob_sum / token_count

    # Calculate perplexity
    perplexity = np.exp(-avg_log_likelihood)

    return log_prob_sum, avg_log_likelihood, perplexity
