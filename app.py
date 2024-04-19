from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

# Load the pre-trained sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# Define API endpoint for similarity prediction
@app.route('/', methods=['POST'])
def predict_similarity():
    # Get text inputs from the request body
    request_data = request.get_json()
    text1 = request_data['text1']
    text2 = request_data['text2']

    # Encode text inputs using the pre-trained model
    embeddings_text1 = model.encode(text1, convert_to_tensor=True).cpu()
    embeddings_text2 = model.encode(text2, convert_to_tensor=True).cpu()

    # Calculate cosine similarity between embeddings
    similarity_score = cosine_similarity(embeddings_text1.reshape(1, -1), embeddings_text2.reshape(1, -1))[0][0]
    
    # Convert similarity score to a JSON serializable format (float)
    similarity_score = float(similarity_score)

    # Return response with similarity score
    response = {'similarity score': similarity_score}
    return jsonify(response)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, port = 8080)
