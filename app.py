from flask import Flask, request, jsonify
import joblib
import logging
import json

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set the logging level to DEBUG

model = joblib.load('./data/xgboost_model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = json.loads(request.data.decode('utf-8'))
        logging.debug(f"Received request: {data}")

        text_content = data.get('text_content').encode('utf-8')
        logging.debug(f"Received request with text_content: {text_content}")

        if not text_content:
            raise ValueError("Missing or empty 'text_content' in the request.")

        prediction = model.predict([text_content])
        readable_prediction = "positive" if prediction[0] == 1 else "negative"

        response_data = {'prediction': readable_prediction}
        logging.debug(f"Predicted result: {response_data}")

        return jsonify(response_data)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000)
