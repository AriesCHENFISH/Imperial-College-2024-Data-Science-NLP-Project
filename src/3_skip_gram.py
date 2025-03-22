import torch
import torch.nn as nn
import torch.optim as optim
import re
import torch.nn.functional as F
import time
from tqdm import tqdm

start_time = time.time()

# vocab_size represents the size of the vocabulary, embed_size represents the dimension of word vectors,
# context_size represents the size of the context (i.e., the length of the considered N-Gram)
class NGramModel(nn.Module):
    def __init__(self, vocab_size, embed_size, context_size):
        super(NGramModel, self).__init__()
        # Create an nn.Embedding layer to map words to word vectors
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        # Create a linear layer to map input context vectors to a hidden layer
        self.linear1 = nn.Linear(context_size * embed_size, 128)
        # Create another linear layer to map the output of the hidden layer to the vocabulary size
        self.linear2 = nn.Linear(128, vocab_size)

    # inputs is a tensor representing the input context
    def forward(self, inputs):
        # Map the input context inputs through the embedding layer self.embeddings,
        # converting each word index to the corresponding word vector.
        # Then, use the view method to reshape the result into a tensor with shape (1, -1),
        # where -1 represents the automatically calculated dimension, maintaining the first dimension as 1.
        embeds = self.embeddings(inputs).view((1, -1))
        # Pass the mapped word vectors embeds to a linear layer self.linear1,
        # applying the ReLU activation function. This linear layer maps the word vectors to a hidden layer.
        out = F.relu(self.linear1(embeds))
        # Pass the output of the hidden layer to another linear layer self.linear2,
        # which maps the output of the hidden layer to the size of the vocabulary,
        # obtaining the original output of the model.
        out = self.linear2(out)
        # Apply LogSoftmax operation to the original output of the model,
        # computing the logarithmic probabilities for each word.
        log_probs = F.log_softmax(out, dim=1)
        # Return the calculated log probabilities as the final output of the model.
        return log_probs

# A function to build a vocabulary from the given text and create mappings from words to indices and indices to words.
# It takes a string parameter text representing the input text content.
def build_vocab(text):
    # Use the regular expression r'\w+' to find all words in the text,
    # where \w+ matches one or more consecutive letters or digits.
    # Convert the text to lowercase for uniform processing.
    words = re.findall(r'\w+', text.lower())
    # Convert the list of found words to a set, removing duplicate words, and obtain the vocabulary vocab.
    vocab = set(words)
    # Create a dictionary word2idx, mapping each word in the vocabulary to a unique index.
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    # Create another dictionary idx2word, mapping indices back to the original words.
    # This dictionary is useful for looking up words based on indices.
    idx2word = {idx: word for word, idx in word2idx.items()}
    return vocab, word2idx, idx2word

# A function to build N-Gram sequences from the given text.
# It takes two parameters: a string text representing the input text content and an integer n representing the length of N-Grams.
def build_ngrams(text, n):
    # Use the regular expression r'\w+' to find all words in the text,
    # where \w+ matches one or more consecutive letters or digits.
    # Convert the text to lowercase for uniform processing.
    words = re.findall(r'\w+', text.lower())
    # Use a list comprehension to generate all N-Grams, forming a list.
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    # Return these N-Gram sequences.
    return ngrams

# A function to train the N-Gram model with a progress bar.
# It takes four parameters: a string text representing the input text content,
# an integer n representing the length of N-Grams, an integer vocab_size representing the size of the vocabulary,
# and a dictionary word2idx representing the mapping from words to indices.
def train_ngram_model_with_progress(text, n, vocab_size, word2idx):
    ngrams = build_ngrams(text, n)
    losses = []
    model = NGramModel(vocab_size, 100, n)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(10):
        total_loss = 0
        # Use tqdm to add a progress bar
        for context_words in tqdm(ngrams, desc=f'Epoch {epoch + 1}/{10}'):
            context_idxs = torch.tensor([word2idx[word] for word in context_words], dtype=torch.long)

            model.zero_grad()
            log_probs = model(context_idxs)
            target = torch.tensor([word2idx[context_words[-1]]], dtype=torch.long)
            loss = loss_function(log_probs, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        losses.append(total_loss)
        print(f'Epoch {epoch + 1}/{10}, Loss: {total_loss}')

    return model

# Read text file
file_path = "pdf_json_2.2.txt"
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Build vocabulary and training data
vocab, word2idx, idx2word = build_vocab(text)
vocab_size = len(vocab)

# Train N-Gram model with progress bar
n = 3
model = train_ngram_model_with_progress(text, n, vocab_size, word2idx)

# Save the model
torch.save(model.state_dict(), 'ngram_model.pth')

# Get trained word vectors
word_vectors = model.embeddings.weight.data.numpy()

# Write words and corresponding vector representations to a text file
with open("3.1new_word_vectors.txt", "w", encoding="utf-8") as file:
    for word, idx in word2idx.items():
        vector = ",".join(str(num) for num in word_vectors[idx])
        file.write(f"{word},{vector}\n")

