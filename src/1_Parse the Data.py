import os
import json
import time

start_time = time.time()
# Set folder path
folder_path = 'D:/pythonProject1/extraction/document_parses/pdf_json'

# Set output file path
output_file_path = 'D:/NLP_python/files/pdf_json_1.txt'

# Traverse all JSON files in the folder
abstract_texts = []

for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as json_file:
            try:
                data = json.load(json_file)

                # Handle the case where the 'abstract' field might be a list
                abstract_list = data.get('abstract', [])

                if isinstance(abstract_list, list):
                    for abstract_item in abstract_list:
                        abstract_text = abstract_item.get('text', '')
                        abstract_texts.append(abstract_text)
                else:
                    abstract_text = abstract_list.get('text', '')
                    abstract_texts.append(abstract_text)

            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {file_path}")

# Write the extracted 'text' field to a new txt file
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for abstract_text in abstract_texts:
        output_file.write(f"{abstract_text}\n")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Segmentation completed in {elapsed_time:.2f} seconds.")
