import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
import numpy as np
from utils import get_frequency_features

# Set device for computation (use GPU if available, else fallback to CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Frequency-Based Model: A classifier based on frequency of characters
class FrequencyClassifier:
    def __init__(self):
        """
        Initialize the FrequencyClassifier that will use Logistic Regression for prediction.
        """
        self.classifier = LogisticRegression(max_iter=1000)

    def fit(self, train_cons_exs, train_vowel_exs):
        """
        Train the classifier using frequency-based features from the consonant and vowel examples.

        :param train_cons_exs: List of training examples for consonants
        :param train_vowel_exs: List of training examples for vowels
        """
        # Generate frequency features for consonant and vowel training examples
        cons_freq_features = get_frequency_features(train_cons_exs)
        vowel_freq_features = get_frequency_features(train_vowel_exs)

        # Combine the feature sets into one training set
        X_train = np.concatenate((cons_freq_features, vowel_freq_features), axis=0)

        # Create labels: 0 for consonants, 1 for vowels
        y_train = np.concatenate((np.zeros(len(cons_freq_features)), np.ones(len(vowel_freq_features))), axis=0)

        # Train the logistic regression model using the frequency-based features
        self.classifier.fit(X_train, y_train)

    def predict(self, text):
        """
        Predict whether a given text is a consonant or vowel based on its frequency features.

        :param text: The input text string to classify
        :return: 0 if consonant, 1 if vowel
        """
        # Generate frequency features for the given text
        features = get_frequency_features([text])

        # Use the trained logistic regression model to predict the class
        return int(self.classifier.predict(features)[0])


# RNN-Based Model: A classifier using a Recurrent Neural Network (RNN) architecture
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        """
        Initialize the RNN classifier with the given dimensions and layers.

        :param vocab_size: The size of the vocabulary (number of unique characters)
        :param embedding_dim: The size of the embedding layer
        :param hidden_dim: The number of units in the hidden layer (LSTM)
        :param output_dim: The number of output classes (2 in our case: consonant or vowel)
        """
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Embedding layer for characters
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # LSTM layer for sequential data
        self.fc = nn.Linear(hidden_dim, output_dim)  # Fully connected layer for classification

    def forward(self, x):
        """
        Forward pass for the RNN model.

        :param x: Input tensor containing text indices
        :return: Output of the model (class probabilities for consonant/vowel)
        """
        embedded = self.embedding(x)  # Pass input through the embedding layer
        rnn_out, _ = self.rnn(embedded)  # Pass through the LSTM layer
        output = self.fc(rnn_out[:, -1, :])  # Use the last hidden state for classification
        return output

    def predict(self, text):
        """
        Predict the class (consonant or vowel) for a given text using the RNN model.

        :param text: The input text string to classify
        :return: 0 for consonant, 1 for vowel
        """
        # Convert text to indices using the vocabulary
        input_indices = torch.tensor([self.vocab[char] for char in text], dtype=torch.long).unsqueeze(0).to(device)

        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation for inference
            output = self(input_indices)  # Pass through the model
            return int(torch.argmax(output, dim=1))  # Return the predicted class


def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    """
    Train the RNN-based classifier using the provided training data.

    :param args: Command line arguments for configurations
    :param train_cons_exs: List of training examples for consonants
    :param train_vowel_exs: List of training examples for vowels
    :param dev_cons_exs: List of development examples for consonants
    :param dev_vowel_exs: List of development examples for vowels
    :param vocab_index: Indexer for mapping characters to indices
    :return: Trained RNN model
    """
    # Set up model parameters
    embedding_dim = 32  # Dimensionality of the embedding layer
    hidden_dim = 74 # Number of units in the LSTM layer
    output_dim = 2  # Output dimensions (2 classes: consonant or vowel)
    batch_size = 74# Batch size for training
    epochs = 10  # Number of training epochs

    # Initialize the RNN model
    vocab_size = len(vocab_index)  # Number of unique characters in the vocabulary
    model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)

    # Prepare training data: Convert each example into tensor of indices
    cons_indices = [torch.tensor([vocab_index.get_index(char) for char in ex], dtype=torch.long) for ex in train_cons_exs]
    vowel_indices = [torch.tensor([vocab_index.get_index(char) for char in ex], dtype=torch.long) for ex in train_vowel_exs]

    # Concatenate the consonant and vowel data
    all_data = cons_indices + vowel_indices
    all_labels = [0] * len(cons_indices) + [1] * len(vowel_indices)

    # Create a DataLoader for batching the training data
    train_data = torch.utils.data.TensorDataset(torch.nn.utils.rnn.pad_sequence(all_data, batch_first=True),
                                                torch.tensor(all_labels, dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification tasks
    optimizer = optim.Adam(model.parameters())  # Adam optimizer for training

    # Train the model over the specified number of epochs
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        for texts, labels in train_loader:
            texts = texts.to(device)  # Move data to the appropriate device (GPU/CPU)
            labels = labels.to(device)  # Move labels to the appropriate device

            optimizer.zero_grad()  # Zero out previous gradients
            outputs = model(texts)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

        print(f"Epoch {epoch+1}/{epochs} completed, Loss: {loss.item()}")

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

class LanguageModel(object):

    def get_log_prob_single(self, next_char, context):
        """
        Scores one character following the given context. That is, returns
        log P(next_char | context)
        The log should be base e
        :param next_char:
        :param context: a single character to score
        :return:
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context):
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return:
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_log_prob_single(self, next_char, context):
        return np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class RNNLanguageModel(LanguageModel):
    def __init__(self, model_emb, model_dec, vocab_index):
        self.model_emb = model_emb
        self.model_dec = model_dec
        self.vocab_index = vocab_index

    def get_log_prob_single(self, next_char, context):
        raise Exception("Implement me")

    def get_log_prob_sequence(self, next_chars, context):
        raise Exception("Implement me")


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    chunk_size = 100
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.learning_rate

    # Initialized RNN model, loss, and optimizer
    model = RNNLanguageModel(args.model_emb, args.model_dec, vocab_index)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Convert train_text to indices based on vocab_index
    train_indices = [vocab_index.index_of(char) for char in train_text]

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        hidden_state = None  # Initialize hidden state

        # Chunk the data and loop over each chunk
        for i in range(0, len(train_indices) - chunk_size, chunk_size):

            chunk = train_indices[i:i + chunk_size]
            input_seq = torch.tensor(chunk[:-1]).unsqueeze(0)  # Input up to the second last character
            target_seq = torch.tensor(chunk[1:]).unsqueeze(0)

            if hidden_state is None:
                hidden_state = model.init_hidden(batch_size)
            else:
                hidden_state = hidden_state.detach()  # Detach hidden state to avoid backprop through entire history

            optimizer.zero_grad()

            # Forward pass through the model
            output, hidden_state = model(input_seq, hidden_state)
            output = output.view(-1, len(vocab_index))
            target_seq = target_seq.view(-1)

            # Calculate loss and perform backpropagation
            loss = criterion(output, target_seq)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    return model
