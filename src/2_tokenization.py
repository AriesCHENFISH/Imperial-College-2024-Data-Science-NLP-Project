import nltk
from nltk.tokenize import sent_tokenize, RegexpTokenizer
import string
import time

# nltk.download('averaged_perceptron_tagger')

def read_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords_list = set(file.read().splitlines())
    return stopwords_list

def nltk_tokenizer(text):
    # Use RegexpTokenizer to exclude punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    return tokens

def process_text(file_content, custom_stop_words, output_file):
    filtered_words_list = []

    for sentence in sent_tokenize(file_content):
        tokens = nltk_tokenizer(sentence)
        filtered_words = [word.lower() for word in tokens if word.lower() not in custom_stop_words]
        filtered_words_list.extend(filtered_words)

    # Write filtered words in larger chunks
    output_file.write(" ".join(filtered_words_list) + "\n")

# Example file paths
stopwords_file_path = r"D:/pythonProject1/extraction/stopwords-en.txt"
input_file_path = r"files/pdf_json_1.txt"
output_file_path = r"files/pdf_json_2.2.txt"

custom_stop_words = read_stopwords(stopwords_file_path)

start_time = time.time()

with open(input_file_path, "r", encoding="utf-8") as input_file, \
     open(output_file_path, "w", encoding="utf-8") as output_file:

    for line in input_file:
        process_text(line, custom_stop_words, output_file)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Segmentation completed in {elapsed_time:.2f} seconds.")
