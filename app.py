from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the pickled model
with open("model.pkl", 'rb') as f:
    model = pickle.load(f)

@app.route('/classify', methods=['POST'])
def classify():
    # Get the message from the request
    message = request.json['message']

    # Perform classification using the loaded model
    is_spam = model.predict([message])[0]

    # Return the classification result as JSON
    return jsonify({'is_spam': bool(is_spam)})

if __name__ == '__main__':
    app.run(debug=True)
