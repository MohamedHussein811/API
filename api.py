from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
from tensorflow.keras.models import load_model
from io import BytesIO
import tempfile
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# Global variable to hold the model instance
loaded_model = None

# Load the Keras model
def load_keras_model():
    global loaded_model
    if loaded_model is None:
        # Download the model file from the provided URL
        url = "https://drive.google.com/uc?export=download&id=1---irB8FRMYQR5uPWHaNB5kzN_Ifn8k7"
        response = requests.get(url)
        model_file = BytesIO(response.content)

        # Save the model to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_model_file:
            temp_model_file.write(model_file.getbuffer())
            temp_model_file_path = temp_model_file.name

        # Load the model
        loaded_model = load_model(temp_model_file_path)
        print("Model loaded")

# Route for accessing model summary
@app.route('/model_summary')
def model_summary():
    load_keras_model()  # Ensure model is loaded
    summary_output = []
    loaded_model.summary(print_fn=lambda x: summary_output.append(x))
    summary_text = "\n".join(summary_output)
    return summary_text

# Route for uploading an image and making predictions
@app.route('/predict', methods=['POST'])
def predict():
    load_keras_model()  # Ensure model is loaded

    # Get the image file from the request
    file = request.files['file']

    # Read image data from file
    image_data = file.read()

    # Convert image data into PIL image
    img = Image.open(BytesIO(image_data))
    img = img.resize((180, 180))  # Resize the image to match model's input size

    # Convert PIL image to numpy array
    img_array = np.array(img)
    img_array = preprocess_input(img_array)  # Apply preprocessing specific to MobileNetV2

    # Reshape the image array to match the expected input shape (1, 180, 180, 3)
    img_array = img_array.reshape(1, 180, 180, 3)

    # Make predictions
    predictions = loaded_model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    print("Predicted class:", predicted_class[0])

    # Example: Return the predicted class as JSON
    return jsonify({'predicted_class': int(predicted_class[0])})


if __name__ == '__main__':
    app.run()
