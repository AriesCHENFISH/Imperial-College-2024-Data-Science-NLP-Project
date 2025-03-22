def read_vectors_from_file(file_path):
    word_vectors = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            word = parts[0]
            vector = [float(val) for val in parts[1:]]
            word_vectors[word] = vector
    return word_vectors

# Read vectors from the text file
file_path = 'pdf_json_3.2true.txt'  # Replace with your file path
word_vectors = read_vectors_from_file(file_path)

import requests

def get_biomedical_entities_from_ols(ontology_id):
    ols_api_url = f'https://www.ebi.ac.uk/ols/api/ontologies/{ontology_id}/terms?size=2000'
    response = requests.get(ols_api_url)

    if response.status_code == 200:
        terms = response.json().get('_embedded', {}).get('terms', [])
        biomedical_entities = [term['label'] for term in terms]
        return biomedical_entities
    else:
        print(f"Error accessing OLS API. Status Code: {response.status_code}")
        return []

# Get a list of biomedical entities
biomedical_entities_from_ols = get_biomedical_entities_from_ols('hp')

from difflib import get_close_matches

def match_biomedical_entities_with_vectors(word_vectors, biomedical_entities, threshold=0.7):
    matched_biomedical_entities = {}

    for entity in biomedical_entities:
        closest_matches = get_close_matches(entity, word_vectors.keys(), n=1, cutoff=threshold)
        if closest_matches:
            matched_biomedical_entities[entity] = {
                'word': closest_matches[0],
                'vector': word_vectors[closest_matches[0]]
            }

    return matched_biomedical_entities

# Match words in the text with biomedical entities
matched_biomedical_entities = match_biomedical_entities_with_vectors(word_vectors, biomedical_entities_from_ols)

output_file_path = 'matched_biomedical_entities.txt'

with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for entity, match_info in matched_biomedical_entities.items():
        output_file.write(f"{match_info['word']}\n")

print(f"Matching results written to: {output_file_path}")

# K-means part

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# File paths
file_path = 'matched_biomedical_entities.txt'

# Read words into an array
word_array = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        word = line.strip().lower()  # Convert to lowercase
        word_array.append(word)

def read_vectors_from_file(file_path):
    word_vectors = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            word = parts[0].lower()  # Convert to lowercase
            vector = [float(val) for val in parts[1:]]
            word_vectors[word] = vector
    print(f"Read {len(word_vectors)} vectors from file.")
    return word_vectors

# Read file containing word vectors
word_vectors_file_path = 'tsne_results.txt'
word_vectors = read_vectors_from_file(word_vectors_file_path)

# Get vectors corresponding to matched words
vectors_for_clustering = np.array([word_vectors[word] for word in word_array])

# Use t-SNE to map high-dimensional vectors to 2D space
tsne = TSNE(n_components=2, perplexity=5, random_state=0)
vectors_2d = tsne.fit_transform(vectors_for_clustering)

# Use the elbow method to determine the optimal number of clusters
def calculate_wcss(data, max_clusters=10):
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss

max_clusters_to_try = 10
wcss_values = calculate_wcss(vectors_for_clustering, max_clusters=max_clusters_to_try)

# Plot the elbow method graph
plt.plot(range(1, max_clusters_to_try + 1), wcss_values, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.show()

# Choose the optimal number of clusters based on the elbow method
optimal_clusters = 4

# Perform clustering using K-means algorithm
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
cluster_labels = kmeans.fit_predict(vectors_for_clustering)

# Plot a scatter plot of the clustering results
plt.figure(figsize=(10, 8))
for i in range(optimal_clusters):
    cluster_points = vectors_2d[cluster_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')

# Add labels
for word, x, y in zip(word_array, vectors_2d[:, 0], vectors_2d[:, 1]):
    plt.annotate(word, (x, y), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

plt.title('t-SNE Visualization of Clusters')
plt.legend()
plt.show()

# Write clustering results to file
output_cluster_file_path = 'cluster_results_new.txt'
with open(output_cluster_file_path, 'w', encoding='utf-8') as output_file:
    for word, cluster_label in zip(word_array, cluster_labels):
        output_file.write(f"Word: {word}, Cluster: {cluster_label}\n")

print(f"Cluster results written to: {output_cluster_file_path}")
