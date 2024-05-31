from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

models = {
    "denseNetModel": None,
    "inceptionV3": None,
    "Xception": None,
    "cnnscratch": None,
    "vgg19":None,
}

model_paths = {
    "denseNetModel": "best_denseNet_model.h5",
    "inceptionV3": "best_inception_V3_model.h5",
    "Xception": "best_Xception_model.h5",
    "cnnscratch": "scratch.h5",
    "vgg19":"best_vgg16_model.h5",}

def load_keras_models():
    global models
    for key, path in model_paths.items():
        if models[key] is None:
            print(f"Loading {key} from {path}")
            models[key] = load_model(path)
            print(f"{key} loaded")

def process_and_predict_image(file, model_key):
    img = Image.open(BytesIO(file.read()))
    img = img.resize((150, 150))
    img_array = preprocess_input(np.array(img))
    img_array = img_array.reshape(1, 150, 150, 3)

    predictions = models[model_key].predict(img_array)

    predicted_class = np.argmax(predictions, axis=1)
    prediction_percentages = (predictions[0] * 100).tolist()

    return predicted_class[0], prediction_percentages

@app.route('/model_summary/<model_key>')
def model_summary(model_key):
    summary_output = []
    models[model_key].summary(print_fn=lambda x: summary_output.append(x))
    return "\n".join(summary_output)

@app.route('/predict/<model_key>', methods=['POST'])
def predict(model_key):
    file = request.files['file']
    predicted_class, prediction_percentages = process_and_predict_image(file, model_key)
    print(model_key)
    return jsonify({
        'predicted_class': int(predicted_class),
        'prediction_percentages': prediction_percentages
    })
if __name__ == '__main__':
    load_keras_models() 
    app.run()
