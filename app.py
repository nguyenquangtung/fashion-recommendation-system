# %%
import tensorflow
from tensorflow import image
from keras.layers import GlobalMaxPooling2D
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
from keras.utils import load_img, img_to_array

model = ResNet50(weights='imagenet', include_top=False,
                 input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

print(model.summary())

# %%
img_path = (
    r'D:\workspace/Recommend_system/fashion-recommendation-system/testdata')


def extract_features(img_path, model):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result


filenames = []

for file in os.listdir(img_path):
    filenames.append(os.path.join(
        img_path, file))

feature_list = []

for file in tqdm(filenames):
    feature_list.append(extract_features(file, model))

pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))

# %%
