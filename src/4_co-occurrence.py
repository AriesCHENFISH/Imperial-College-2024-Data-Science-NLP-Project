import numpy as np
from scipy.sparse import dok_matrix
from collections import Counter
from tqdm import tqdm
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Read segmented text data from a text file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        texts = file.readlines()
    return texts

# Build co-occurrence matrix with progress bar
def build_cooccurrence_matrix(texts):
    vocab = set()
    for text in texts:
        words = text.split()
        vocab.update(words)

    vocab = list(vocab)
    vocab_indices = {word: index for index, word in enumerate(vocab)}

    cooccurrence_matrix = dok_matrix((len(vocab), len(vocab)), dtype=np.float64)

    for text in tqdm(texts, desc="Building Co-occurrence Matrix"):
        words = text.split()
        word_indices = [vocab_indices[word] for word in words if word in vocab]
        for i in range(len(word_indices)):
            for j in range(i + 1, len(word_indices)):
                cooccurrence_matrix[word_indices[i], word_indices[j]] += 1
                cooccurrence_matrix[word_indices[j], word_indices[i]] += 1

    return cooccurrence_matrix, vocab_indices

# Main function
def main():
    # Read text data
    file_path = 'D:\pythonProject1\extraction\pdf_json_2_100percent.txt'
    texts = read_text_file(file_path)
    print("1")
    # Build co-occurrence matrix
    cooccurrence_matrix, vocab_indices = build_cooccurrence_matrix(texts)
    print("2")
    # Optionally: Normalize the co-occurrence matrix
    row_sums = np.array(cooccurrence_matrix.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    cooccurrence_matrix_normalized = cooccurrence_matrix.tocsr() / row_sums[:, np.newaxis]


    # In addition to the existing code

    # Find words co-occurring with "COVID19" based on cooccurrence_matrix and vocab_indices
    covid_related_entities = ['COVID19']
    cooccurrence_counts = Counter()

    for entity in covid_related_entities:
        if entity in vocab_indices:
            index = vocab_indices[entity]
            cooccurrence_counts[entity] = cooccurrence_matrix[index, :].sum()

    # Sort by co-occurrence frequency
    sorted_entities = sorted(cooccurrence_counts.items(), key=lambda x: x[1], reverse=True)



if __name__ == "__main__":
    main()
