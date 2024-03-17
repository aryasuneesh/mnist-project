from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('svm_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.json['data']

    # Perform prediction
    prediction = model.predict(data)

    # Return prediction
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
