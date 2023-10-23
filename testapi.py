from flask import Flask, request, render_template, jsonify
import requests
import os
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
import mysql.connector

# global variable
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False,
                 input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# database connection details
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "07032001",
    "database": "fashionstorewebsite",
}


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


app = Flask(__name__)


# @app.route('/')
# def index():
#     return render_template('URLSubmit.html')


# @app.route('/download_image', methods=['POST'])
# def download_image():
#     # Lấy địa chỉ URL hình ảnh từ dữ liệu POST
#     image_url = request.form['url']

#     if image_url:
#         # Gửi yêu cầu HTTP để lấy hình ảnh
#         response = requests.get(image_url)

#         if response.status_code == 200:
#             # Tạo thư mục để lưu hình ảnh nếu nó chưa tồn tại
#             if not os.path.exists("downloaded_images"):
#                 os.makedirs("downloaded_images")

#             # Tạo tên tệp cho hình ảnh đã tải
#             filename = os.path.join("downloaded_images", "image.jpg")

#             # Lưu hình ảnh xuống ổ đĩa
#             with open(filename, "wb") as file:
#                 file.write(response.content)

#             return jsonify({"message": "Save successfully."})
#         else:
#             return jsonify({"error": "Download request failed."})

#     else:
#         return jsonify({"error": "Image URL not provided."})


product_name = "1-red"


@app.route('/recommendResults', methods=['GET'])
def recommendResults():
    # Lấy tên sản phẩm từ yêu cầu
    product_name = request.args.get('product_name')
    if not product_name:
        return jsonify({"error": "Missing product_name parameter"})
    else:
        # product_name = "1-white.jpg" #for test api
        product_name_path = os.path.abspath(r'dataset/'+product_name+'.jpg')
        # feature extract
        # save_uploaded_file(uploadimg)
        features = feature_extraction(product_name_path, model)
        # st.text(features)

        # recommendention
        indices = recommend(features, feature_list)
        recommendResults = []
        try:
            # Connect to the MySQL database
            connection = mysql.connector.connect(**db_config)
            if connection.is_connected():
                cursor = connection.cursor()
                for i in range(1, 6):
                    image_url = (filenames[indices[0][i]])
                    file_name = os.path.basename(image_url)
                    parts = file_name.split('-')
                    if len(parts) == 2:
                        id, fullpath = parts
                        # Lấy phần trước dấu chấm
                        color = fullpath.split('.')[0]
                    # Truy vấn SQL SELECT để lấy các sản phẩm có ID và màu tương ứng
                        cursor.execute(
                            "SELECT id,product_id,name,selling_price,discount,brand,size,color,available_quantity,image_1,image_2,image_3,image_4,overall_rating FROM fashionstorewebsite.product_info_for_ui WHERE product_id = %s AND color = %s", (
                                id, color)
                        )
                        product = cursor.fetchone()
                        if product:
                            # create a distionary product for showing the key value
                            product_info = {
                                "id": product[0],
                                "product_id": product[1],
                                "name": product[2],
                                "sellingPrice": product[3],
                                "discount": product[4],
                                "brand": product[5],
                                "size": product[6],
                                "color": product[7],
                                "availableQuantity": product[8],
                                "image1": product[9],
                                "image2": product[10],
                                "image3": product[11],
                                "image4": product[12],
                                "overallRating": product[13]
                            }
                            # list key ordered
                            key_order = ["id", "product_id", "name", "sellingPrice", "discount", "size",
                                         "color", "availableQuantity", "image1", "image2", "image3", "image4", "overallRating"]
                            # Sort data following key order
                            sorted_data = {
                                key: product_info[key] for key in key_order}
                            recommendResults.append(sorted_data)
        except mysql.connector.Error as e:
            print(f"Error: {e}")
        finally:
            if 'connection' in locals() and connection.is_connected():
                cursor.close()
                connection.close()
                print("MySQL connection is closed")
        # If can not find the product by product name
        if not recommendResults:
            return jsonify({"error": "No products found for the given product_name"})
        else:
            # Declare a dictionary has results and content
            response_data = {"result": "success", "content": recommendResults}
            # Make sure "results" apear before "content" in dictionary
            response_data = {
                "result": response_data["result"], "content": response_data["content"]}
            # return jsonify({"""content""": recommendResults})
            print(response_data)
            return jsonify(response_data)


if __name__ == '__main__':
    app.run(debug=True)
