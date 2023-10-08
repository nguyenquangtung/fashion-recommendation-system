from flask import Flask, jsonify, request, send_file,  render_template
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


feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False,
                 input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


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
        n_neighbors=7, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices


# steps

# uploaded_file = st.file_uploader("Choose an image")
# if uploaded_file is not None:
#     if save_uploaded_file(uploaded_file):
#         # display the file
#         display_image = Image.open(uploaded_file)
#         st.image(display_image)
#         # feature extract
#         features = feature_extraction(os.path.join(
#             "uploads", uploaded_file.name), model)
#         # st.text(features)
#         # recommendention
#         indices = recommend(features, feature_list)
#         print(indices)
#         # show
#         col1, col2, col3, col4, col5 = st.columns(5)

#         with col1:
#             st.image(Image.open(filenames[indices[0][2]]))

#         with col2:
#             st.image(Image.open(filenames[indices[0][3]]))
#         with col3:
#             st.image(Image.open(filenames[indices[0][4]]))
#         with col4:
#             st.image(Image.open(filenames[indices[0][5]]))
#         with col5:
#             st.image(Image.open(filenames[indices[0][6]]))
#     else:
#         st.header("Some error occured in file upload")

uploadImg = r'D:\workspace\Recommend_system\fashion-recommendation-system\testdata\1163.jpg'
# feature extract
# save_uploaded_file(uploadimg)
features = feature_extraction(uploadImg, model)
# st.text(features)
# recommendention
indices = recommend(features, feature_list)

app = Flask(__name__)


@app.route('/recommendResults', methods=['GET'])
def recommendResults():
    result_images = []
    for i in range(1, 6):
        image_url = (filenames[indices[0][i]])
        result_images.append(image_url)
    return jsonify({"images": result_images})
    # # Trả về trang HTML với danh sách đường dẫn hình ảnh (chưa chạy đc)
    # return render_template('fashion-recommendation-system\front-end\RecommendImages.html', images=result_images[0])


if __name__ == '__main__':
    app.run(debug=True, port=8000)
