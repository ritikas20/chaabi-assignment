# Requirements
flask

pandas

transformers

scikit-learn

numpy

qdrant-client

openai

os

llama-index cohere datasets pandas
'rich[jupyter]'

# Setup Instructions:

**Install Required Packages:**

Install Python (if not already installed).

**Create a virtual environment (optional but recommended):**

python -m venv myenv

**Activate the virtual environment (if created):**

source myenv/bin/activate  # for Linux/Mac
myenv\Scripts\activate     # for Windows

**Install required packages using pip:**

pip install pandas transformers scikit-learn numpy qdrant-client flask openai

**Download BERT Model Weights:**

The code uses the 'bert-base-uncased' model. It will automatically download the model weights the first time you run the script that initializes the BERT model and tokenizer.

**Set Up Environment Variables:**

Before running the code, ensure you've set the environment variables for OpenAI, Cohere, and Qdrant. Replace <YOUR_OPENAI_API_KEY>, <YOUR_COHERE_API_KEY>, and <YOUR_QDRANT_API_KEY> with your actual API keys.

export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
export COHERE_API_KEY=<YOUR_COHERE_API_KEY>
export QDRANT_API_KEY=<YOUR_QDRANT_API_KEY>

**Prepare the CSV File:**

Ensure you have the CSV file (bigBasketProducts.csv) located at the specified path.
Adjust the path if necessary or ensure the CSV file exists in the mentioned location.

**Run the Flask App:**

Run the Flask app using the provided script.

# Output of sample query:

**Please refer to the attached image**

write your query: write details of Garlic Oil - Vegetarian Capsule 500 mg

Answer:

The Garlic Oil - Vegetarian Capsule 500 mg is a product under the Beauty & Hygiene category and specifically falls under the Hair Care sub-category. It is a product by the brand Sri Sri Ayurveda. The sale price of this product is 220 INR, while the market price is also 220 INR.

This capsule contains Garlic Oil, which offers several health benefits. It is known to aid in proper digestion, help maintain optimum cholesterol levels, support cardiovascular health, and also help in building immunity. The capsule is suitable for vegetarians.

For more beauty tips, tricks, and information, you can visit the website https://bigbasket.blog/.
