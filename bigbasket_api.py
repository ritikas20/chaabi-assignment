import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter
from flask import Flask, request, jsonify
import openai
import os

# Initialize Flask app
app = Flask(__name__)

os.environ['OPENAI_API_KEY'] = "<YOUR_OPENAI_API_KEY>"
os.environ['COHERE_API_KEY'] = "<YOUR_COHERE_API_KEY>"
os.environ['QDRANT_API_KEY'] = "<YOUR_QDRANT_API_KEY>"


def check_environment_keys():
    """
    Utility Function that you have the NECESSARY Keys
    """
    if os.environ.get('OPENAI_API_KEY') is None:
        raise ValueError(
            "OPENAI_API_KEY cannot be None. Set the key using os.environ['OPENAI_API_KEY']='sk-xxx'"
        )
    if os.environ.get('COHERE_API_KEY') is None:
        raise ValueError(
            "COHERE_API_KEY cannot be None. Set the key using os.environ['COHERE_API_KEY']='xxx'"
        )
    if os.environ.get("QDRANT_API_KEY") is None:
        print("[Optional] If you want to use the Qdrant Cloud, please get the Qdrant Cloud API Keys and URL")


check_environment_keys()

# Load dataset
import pandas as pd

# Correct file path (use the absolute path to your CSV file)
df = pd.read_csv(r'c:/Users/tanis/Downloads/bigBasketProducts.csv')

# Convert descriptions to string to avoid errors in tokenization
df['description'] = df['description'].astype(str)

# Initialize BERT model and tokenizer for embeddings
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to generate vector embeddings using BERT
def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Function to store embeddings in Qdrant DB
def store_embeddings_in_qdrant(embeddings, qdrant_client, collection_name='products'):
    for idx, embedding in enumerate(embeddings):
        qdrant_client.replace(collection_name=collection_name, point_id=idx, vector=embedding.tolist(), payload=df.iloc[idx].to_dict())

# Initialize Qdrant client
qdrant_client = QdrantClient(host='localhost', port=6333)

# Process and store embeddings
embeddings = np.array([generate_embeddings(text) for text in df['description']])
store_embeddings_in_qdrant(embeddings, qdrant_client)

# Function to implement LLM for Q/A using OpenAI's GPT-3
def process_query_with_llm(query):
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=query,
      max_tokens=100
    )
    return response.choices[0].text.strip()

# API endpoint for querying
@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query = data.get('query', '')
    # Process the query through LLM and retrieve data from Qdrant
    response = process_query_with_llm(query)
    return jsonify({'response': response})

# Main function to run the Flask app
if __name__ == '__main__':
    app.run(debug=True)



    