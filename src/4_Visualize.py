# TODO: add your solution
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar

# From txt file, load word representations where each line consists of a word and its corresponding vector representation
with open('files/pdf_json_3.2true.txt', 'r', encoding='utf-8') as file:
    word_vectors = [line.strip().split() for line in file]

# Extract words and vector representations
words, vectors = zip(*[(line[0], list(map(float, line[1:]))) for line in word_vectors])

# Convert lists to numpy arrays
vectors = np.array(vectors)

# Choose t-SNE parameters
tsne = TSNE(n_components=2, random_state=42)

# Use t-SNE for dimensionality reduction
word_tsne = np.zeros((len(words), 2))  # Initialize array to store t-SNE results
for i in tqdm(range(0, len(words), 1000), desc="t-SNE Progress"):
    end_idx = min(i + 1000, len(words))
    word_tsne[i:end_idx] = tsne.fit_transform(vectors[i:end_idx])

with open('files/tsne_results_1.txt', 'w', encoding='utf-8') as tsne_file:
    for i, word in enumerate(words):
        tsne_file.write(f"{word} {word_tsne[i, 0]} {word_tsne[i, 1]}\n")

# Visualize the results without labels, with smaller points, and semi-transparent colors
plt.figure(figsize=(10, 8))
plt.scatter(word_tsne[:, 0], word_tsne[:, 1], alpha=0.5, s=10)  # Adjust alpha for transparency and s for point size

plt.title('t-SNE Visualization of Word Representations')
plt.show()
