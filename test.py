# %%
import pickle
import tensorflow
import numpy as np
from numpy.linalg import norm
from tensorflow import image
from keras.layers import GlobalMaxPooling2D
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input
from sklearn.neighbors import NearestNeighbors
from keras.utils import load_img, img_to_array
import cv2

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False,
                 input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

img = load_img(
    r'D:\workspace\Recommend_system\fashion-recommendation-system\testdata\1163.jpg', target_size=(224, 224))
img_array = img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

neighbors = NearestNeighbors(
    n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)

distances, indices = neighbors.kneighbors([normalized_result])

print(indices)

for file in indices[0][1:5]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output', cv2.resize(temp_img, (512, 512)))
    cv2.waitKey(0)

# %%
