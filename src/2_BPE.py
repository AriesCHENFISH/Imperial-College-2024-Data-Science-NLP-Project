from tokenizers import ByteLevelBPETokenizer
import re
import time
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def bpe_tokenizer(text, vocab_size=30000):
    """
    Tokenize the input text using Byte-Pair Encoding (BPE).

    Parameters:
    text (str): Input text to be tokenized.
    vocab_size (int): Vocabulary size for BPE.

    Returns:
    list: A list of tokens.
    """
    # Initialize the BPE tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Train the BPE tokenizer
    tokenizer.train_from_iterator([text], vocab_size=vocab_size)

    # Tokenize using the trained BPE tokenizer
    encoded = tokenizer.encode(text)
    tokens = encoded.tokens

    return tokens

# Read the txt file containing abstracts
input_file_path = r"C:\Users\think\PycharmProjects\pythonProject2\abstracts1.txt"
output_file_path = r"C:\Users\think\PycharmProjects\pythonProject2\2.3output.txt"

start_time = time.time()

with open(input_file_path, "r", encoding="utf-8") as input_file:
    # Read the entire file content
    file_content = input_file.read()

# Remove all symbols, keeping only words
file_content = re.sub(r'[^\w\s]', '', file_content)

# Tokenize using the BPE tokenizer
tokens = bpe_tokenizer(file_content)

# Use NLTK for part-of-speech tagging, keeping only nouns
tagged_tokens = nltk.pos_tag(word_tokenize(file_content))
noun_tokens = [token[0] for token in tagged_tokens if token[1] in ['NN', 'NNS', 'NNP', 'NNPS']]

with open(output_file_path, "w", encoding="utf-8") as output_file:
    # Write nouns to the file, separated by spaces
    output_file.write(" ".join(noun_tokens))

# Tokenization results have been saved to a new file
elapsed_time = time.time() - start_time
print(f"Tokenization completed in {elapsed_time:.2f} seconds.")
