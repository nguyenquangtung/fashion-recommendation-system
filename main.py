'''
Main for testing model on streamlit interface
'''
# %% Run test on Streamlit
import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from keras.layers import GlobalMaxPooling2D
from keras.applications import ResNet50
from utils import func


feature_list = np.array(pickle.load(open('.\\dataloader\\embeddings.pkl', 'rb')))
filenames = pickle.load(open('.\\dataloader\\filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False,
                 input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

# steps

uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if func.save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        # feature extract
        features = func.feature_extraction(os.path.join(
            "uploads", uploaded_file.name), model)
        # st.text(features)
        # recommendention
        indices = func.recommend(features, feature_list)
        print(indices)
        # show
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(Image.open(filenames[indices[0][2]]))

        with col2:
            st.image(Image.open(filenames[indices[0][3]]))
        with col3:
            st.image(Image.open(filenames[indices[0][4]]))
        with col4:
            st.image(Image.open(filenames[indices[0][5]]))
        with col5:
            st.image(Image.open(filenames[indices[0][6]]))
    else:
        st.header("Some error occured in file upload")

# %%
