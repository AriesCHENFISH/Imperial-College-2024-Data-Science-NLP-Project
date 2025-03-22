from transformers import BertTokenizer

# Read the text file
with open('pdf_json_2.2.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Load the BertTokenizer
model_name = "D:/pythonProject1/extraction/Bert/biobert_v1.1_pubmed/model.ckpt-1000000.index"
tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)

# Process the text using BertTokenizer
tokenized_text = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))

# Print the processed results
print(tokenized_text)
