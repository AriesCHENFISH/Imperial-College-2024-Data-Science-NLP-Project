from tqdm import tqdm
import time

def initialize_vocab(data):
    vocab = set()
    for word in data:
        vocab.update(list(word))
    return vocab

def get_stats(data):
    stats = {}
    for word in data:
        symbols = list(word)
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            stats[pair] = stats.get(pair, 0) + 1
    return stats

def merge_vocab(pair, vocab):
    new_vocab = set()
    bigram = ''.join(pair)
    replacement = ''.join(pair)
    for word in vocab:
        new_word = word.replace(bigram, replacement)
        new_vocab.add(new_word)
    return new_vocab

def learn_bpe(data, num_merges, progress_bar=True):
    vocab = initialize_vocab(data)

    start_time = time.time()
    progress_bar = tqdm(range(num_merges)) if progress_bar else range(num_merges)
    for _ in progress_bar:
        stats = get_stats(data)
        best_pair = max(stats, key=stats.get)
        vocab = merge_vocab(best_pair, vocab)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'BPE model trained in {elapsed_time:.2f} seconds')

    return vocab

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read().splitlines()
    return data

import re
from nltk import word_tokenize, pos_tag

def filter_nouns_and_remove_non_english(text):
    filtered_data = []
    for sentence in text:
        # Remove non-English characters
        english_text = re.sub(r'[^a-zA-Z\s]', '', sentence)

        # Tokenize and POS tag
        tokens = word_tokenize(english_text)
        tagged = pos_tag(tokens)

        # Keep nouns and convert to lowercase
        nouns = [word.lower() for word, pos in tagged if pos.startswith('N')]

        # Reassemble the sentence
        filtered_sentence = ' '.join(nouns)
        filtered_data.append(filtered_sentence)
    return filtered_data

# Read text file
biomedical_data = read_text_file(r"pdf_json_1.txt")

# Keep only nouns and remove symbols, numbers, and special characters
filtered_data = filter_nouns_and_remove_non_english(biomedical_data)

biomedical_bpe_vocab = learn_bpe(filtered_data, num_merges=100, progress_bar=True)

# Save biomedical BPE vocabulary to file
with open('biomedical_bpe_vocab.txt', 'w', encoding='utf-8') as file:
    for token in biomedical_bpe_vocab:
        file.write(token + '\n')
