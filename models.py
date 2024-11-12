import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from utils import get_frequency_features


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


def train_lm(args, train_text, dev_text, vocab_index): #training
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
