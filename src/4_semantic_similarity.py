import numpy as np
from tqdm import tqdm
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def load_vectors_from_txt(file_path):
    # Load embedding vectors from a txt file
    vectors = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(' ')
            entity = parts[0]
            vector = np.array([float(val) for val in parts[1:]])
            vectors[entity] = vector
    return vectors


def cosine_similarity(vec1, vec2):
    # Calculate cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity


def generate_wordcloud(sorted_entities):
    wordcloud_text = {entity: float(score) for entity, score in sorted_entities}

    # Create WordCloud object
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wordcloud_text)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


def main():
    # File paths
    vectors_file_path = 'D:\pythonProject1\extraction\pdf_json_3.2true.txt'  # Replace with your file path
    output_file_path = 'output_similarity.txt'

    # Load embedding vectors with replacements
    loaded_vectors = load_vectors_from_txt(vectors_file_path)

    # Extract the vector for corona
    corona_vector = loaded_vectors.get('corona')

    if corona_vector is not None:
        print("Successfully extracted the vector for corona")
    else:
        print("Vector for corona not found")

    # Calculate similarity scores with progress bar
    similarity_scores = {}
    for entity, entity_vector in tqdm(loaded_vectors.items(), desc="Calculating Similarity Scores"):
        similarity_scores[entity] = cosine_similarity(corona_vector, entity_vector)

    # Sort the results
    sorted_entities = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

    # Output results to a new txt file
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for entity, similarity_score in sorted_entities:
            output_file.write(f"{entity}: {similarity_score}\n")

    # Generate and display the word cloud
    generate_wordcloud(sorted_entities)


if __name__ == "__main__":
    main()
