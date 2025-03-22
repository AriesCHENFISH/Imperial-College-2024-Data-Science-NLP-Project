import torch
import torch.nn as nn
import torch.optim as optim
import re
import torch.nn.functional as F
from tqdm import tqdm

# Download NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# A class defining the NGram language model
class NGramModel(nn.Module):
    def __init__(self, vocab_size, embed_size, context_size):
        super(NGramModel, self).__init__()
        # Create an nn.Embedding layer to map words to word vectors
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        # Create a linear layer to map input context vectors to a hidden layer
        self.linear1 = nn.Linear(context_size * embed_size, 128)
        # Create another linear layer to map the output of the hidden layer to the vocabulary size
        self.linear2 = nn.Linear(128, vocab_size)

    # Forward method to define the forward pass of the model
    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

# Build vocabulary from given text
def build_vocab(text):
    words = re.findall(r'\w+', text.lower())
    vocab = set(words)
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return vocab, word2idx, idx2word

# Build N-Gram sequences from given text
def build_ngrams(text, n):
    words = re.findall(r'\w+', text.lower())
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return ngrams

# Train the N-Gram model
def train_ngram_model(text, n, vocab_size, word2idx, num_epoch, lr, device, weight_decay):
    ngrams = build_ngrams(text, n)
    losses = []

    model = NGramModel(vocab_size, 100, n)
    model.to(device)
    model.train()

    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    ngrams_idxs = []
    for context_words in ngrams:
        context_idxs = torch.tensor([word2idx[word] for word in context_words], dtype=torch.long).to(device)
        ngrams_idxs.append(context_idxs)

    for epoch in range(num_epoch):
        total_loss = 0
        for i in tqdm(range(len(ngrams)), desc=f'Epoch {epoch + 1}/{num_epoch}', unit='batch'):
            context_words = ngrams[i]
            context_idxs = ngrams_idxs[i].to(device)

            model.zero_grad()
            log_probs = model(context_idxs)
            target = torch.tensor([word2idx[context_words[-1]]], dtype=torch.long).to(device)
            loss = loss_function(log_probs, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        losses.append(total_loss)

        torch.save(model.state_dict(), f'ngram_model_ep{epoch}.pth')
        print(f'Epoch {epoch + 1}/{num_epoch}, Loss: {total_loss}')

    return model

if __name__ == '__main__':
    # Determine the current device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    num_epoch = 10
    lr = 1e-2
    n = 3
    weight_decay = 0

    # Read text file
    file_path = "pdf_json_2.2.txt"
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Build vocabulary and training data
    vocab, word2idx, idx2word = build_vocab(text)
    vocab_size = len(vocab)

    # Train N-Gram model
    model = train_ngram_model(text, n, vocab_size, word2idx, num_epoch, lr, device, weight_decay)

    # Save the model
    torch.save(model.state_dict(), 'ngram_model.pth')

    # Get trained word vectors
    word_vectors = model.embeddings.weight.data.numpy()

    # Write words and corresponding vector representations to a text file
    with open("3.1new_word_vectors.txt", "w", encoding="utf-8") as file:
        for word, idx in word2idx.items():
            vector = ",".join(str(num) for num in word_vectors[idx])
            file.write(f"{word},{vector}\n")
