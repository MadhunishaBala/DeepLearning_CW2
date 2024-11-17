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
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1

class RNNClassifier(ConsonantVowelClassifier, nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, output_dim, vocab_index, num_layers, dropout, bidirectional):
        super(RNNClassifier, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers, dropout=dropout,bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)
        self.vocab_index = vocab_index

    def forward(self, context_tensor):
        embedding = self.emb(context_tensor) #Embedding for input
        _, hidden = self.gru(embedding)  # Output hidden state of GRU
        if isinstance(hidden, tuple):  # For GRU with num_layers > 1
            hidden = hidden[0]
        output = self.fc(hidden[-1])
        return output

    def predict(self, context):# Convert context string to indices
        context_indices = torch.tensor(
            [self.vocab_index.index_of(char) for char in context],
            dtype=torch.long
        ).unsqueeze(0)  # Add batch dimension
        output = self.forward(context_indices)
        prediction = torch.argmax(output, dim=1).item()
        return prediction

def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)

def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index): #args wasn't used cause the parameters are intialized.
    emb_dim = 50
    hidden_dim = 50
    learning_rate = 0.001
    num_epochs = 10
    output_dim = 2
    batch_size = 64
    num_layers = 2
    dropout = 0.5
    bidirectional = False
    vocab_size = len(vocab_index)

    # Initialize the model
    model = RNNClassifier(vocab_size, emb_dim, hidden_dim, output_dim, vocab_index,num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)

    # Training data
    train_data = [(ex, 0) for ex in train_cons_exs] + [(ex, 1) for ex in train_vowel_exs]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def get_batches(data, batch_size): # Convert data to batches
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

            _, predicted = torch.max(output, 1)
            correct += (predicted == label_tensor).sum().item()
            total += label_tensor.size(0)

        #Training accuracy
        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}, Training Accuracy: {accuracy:.2f}%")

        # Evaluation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for context, label in zip(dev_cons_exs + dev_vowel_exs, [0] * len(dev_cons_exs) + [1] * len(dev_vowel_exs)):
                context_indices = torch.tensor([vocab_index.index_of(char) for char in context], dtype=torch.long).unsqueeze(0)
                output = model(context_indices)

                _, predicted = torch.max(output, 1)

                # Convert label to a tensor
                label_tensor = torch.tensor([label], dtype=torch.long)
                correct += (predicted == label_tensor).sum().item()
                total += label_tensor.size(0)  # Now label_tensor is a tensor, so .size(0) works

            # Testing Accuracy
            test_accuracy = 100 * correct / total
            print(f"Testing Accuracy: {test_accuracy:.2f}%")

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


class RNNLanguageModel(nn.Module):  # Inherit from nn.Module
    def __init__(self, emb_dim, hidden_dim, vocab_index):
        super(RNNLanguageModel, self).__init__()
        self.vocab_size = len(vocab_index)
        self.embedding = nn.Embedding(self.vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, self.vocab_size)
        self.vocab_index = vocab_index

    def forward(self, input_seq, hidden_state):
        embedded = self.embedding(input_seq)
        rnn_out, hidden_state = self.rnn(embedded, hidden_state)
        output = self.fc(rnn_out)
        return output, hidden_state

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.rnn.hidden_size),
                torch.zeros(1, batch_size, self.rnn.hidden_size))

    def get_log_prob_single(self, next_char, context):
        context_indices = [self.vocab_index.index_of(char) for char in context]
        context_tensor = torch.tensor(context_indices, dtype=torch.long).unsqueeze(0)
        next_char_index = self.vocab_index.index_of(next_char)
        with torch.no_grad():
            hidden_state = self.init_hidden(1)
            output, _ = self.forward(context_tensor, hidden_state)
            output = output[:, -1, :]
        log_probs = torch.log_softmax(output, dim=1)
        return log_probs[0, next_char_index].item()

    def get_log_prob_sequence(self, next_chars, context):
        total_log_prob = 0.0
        current_context = context
        for char in next_chars:
            log_prob = self.get_log_prob_single(char, current_context)
            total_log_prob += log_prob
            current_context += char
        return total_log_prob




def train_lm(args, train_text, dev_text, vocab_index):
    emb_dim = 50
    hidden_dim = 128
    chunk_size = 100
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001

    # Initialize the model, optimizer, and loss criterion
    model = RNNLanguageModel(emb_dim, hidden_dim, vocab_index)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Convert train text to indices
    train_indices = [vocab_index.index_of(c) for c in train_text]
    dev_indices = [vocab_index.index_of(c) for c in dev_text]

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        hidden_state = model.init_hidden(batch_size)

        # Training loop
        for i in range(0, len(train_indices) - chunk_size, chunk_size):
            chunk = train_indices[i:i + chunk_size]
            input_seq = torch.tensor(chunk[:-1]).unsqueeze(0).repeat(batch_size, 1)  # [batch_size, seq_len]
            target_seq = torch.tensor(chunk[1:]).unsqueeze(0).repeat(batch_size, 1)  # [batch_size, seq_len]

            hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())  # Detach to prevent backprop through history
            optimizer.zero_grad()

            # Forward pass
            output, hidden_state = model(input_seq, hidden_state)

            # Reshape for CrossEntropyLoss
            loss = criterion(output.view(-1, model.vocab_size), target_seq.view(-1))  # Flatten output and target
            loss.backward()
            optimizer.step()

        # After each epoch, evaluate on the development set
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            hidden_state = model.init_hidden(batch_size)
            dev_loss = 0.0
            for i in range(0, len(dev_indices) - chunk_size, chunk_size):
                chunk = dev_indices[i:i + chunk_size]
                input_seq = torch.tensor(chunk[:-1]).unsqueeze(0).repeat(batch_size, 1)  # [batch_size, seq_len]
                target_seq = torch.tensor(chunk[1:]).unsqueeze(0).repeat(batch_size, 1)  # [batch_size, seq_len]

                output, hidden_state = model(input_seq, hidden_state)
                loss = criterion(output.view(-1, model.vocab_size), target_seq.view(-1))  # Flatten output and target
                dev_loss += loss.item()

            dev_loss /= (len(dev_indices) // chunk_size)  # Average loss over all chunks in dev set
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss.item():.4f}, Dev Loss: {dev_loss:.4f}")

    return model






