from flask import Flask, request, jsonify
import os
import pickle
import numpy as np
import tensorflow as tf
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow import image
from keras.layers import GlobalMaxPooling2D
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from keras.utils import load_img, img_to_array
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from PIL import Image

app = Flask(__name__)


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


def feature_extraction(img_path, model):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result


def recommend(features, feature_list):
    neighbors = NearestNeighbors(
        n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices


feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False,
                 input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    tf.keras.layers.GlobalMaxPooling2D()
])

app = Flask(__name__)


@app.route("/")
@app.route('/recommend', methods=['POST'])
def recommend_api():
    try:
        # Get the uploaded file from the POST request
        uploaded_file = request.files['file']
        if uploaded_file:
            # Save the uploaded file
            save_uploaded_file(uploaded_file)
            # Feature extraction
            features = feature_extraction(os.path.join(
                "uploads", uploaded_file.filename), model)
            # Recommendation
            indices = recommend(features, feature_list)
            return jsonify({"indices": indices[0].tolist()})
        else:
            return jsonify({"error": "No file uploaded."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
    # import requests

    # # Gửi hình ảnh lên API
    # files = {'file': ('image.jpg', open('image.jpg', 'rb'))}
    # response = requests.post('http://localhost:5000/recommend', files=files)

    # if response.status_code == 200:
    #     recommendations = response.json()
    #     print("Recommendations:")
    #     for rec in recommendations:
    #         print(rec)
    # else:
    #     print("Error:", response.json())
