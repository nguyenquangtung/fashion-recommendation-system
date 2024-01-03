"""
API use for e-commerce services
"""
import os
import pickle
from flask import Flask, jsonify, request
import numpy as np
import mysql.connector
from flask_cors import CORS
from configs import config
from utils import func
from model import fashion_model


# global variable
feature_list = np.array(pickle.load(open('dataloader/embeddings.pkl', 'rb')))
filenames = pickle.load(open('dataloader/filenames.pkl', 'rb'))

model = fashion_model.FashionRecommendationModel().model


# database connection details
DB = config.DBconfig()

app = Flask(__name__)
CORS(app)


@app.route("/")
def hello():
    message = "Hello, this is default page of API recommender system for Fashion Store"
    return message


@app.route("/recommendResults", methods=["POST"])
def recommendResults():
    if request.is_json:
        try:
            # Trích xuất dữ liệu JSON từ yêu cầu
            data = request.get_json()
            product_name = data.get("product")

            # Kiểm tra xem 'product' có tồn tại trong JSON không
            if not product_name:
                return jsonify({"error": "Missing product_name parameter"})
            else:
                product_name_path = os.path.abspath(
                    os.path.join("dataset", product_name + ".jpg")
                )
                # feature extract
                # save_uploaded_file(uploadimg)
                features = func.feature_extraction(product_name_path, model)
                # st.text(features)

                # recommendention
                indices = func.recommend(features, feature_list)
                recommendResults = []
                try:
                    # Connect to the MySQL database
                    connection = mysql.connector.connect(**DB._db_config)
                    if connection.is_connected():
                        cursor = connection.cursor()
                        for i in range(2, 7):
                            image_url = filenames[indices[0][i]]
                            file_name = os.path.basename(image_url)
                            parts = file_name.split("-")
                            if len(parts) == 2:
                                id, fullpath = parts
                                # Lấy phần trước dấu chấm
                                color = fullpath.split(".")[0]
                                if ('dataset\\' in id):
                                    id = id.split('\\')[1]
                                # Truy vấn SQL SELECT để lấy các sản phẩm có ID và màu tương ứng
                                cursor.execute(
                                    "SELECT id,product_id,name,selling_price,discount,brand,size,color,available_quantity,image_1,image_2,image_3,image_4,overall_rating FROM fashionstorewebsite.product_info_for_ui WHERE product_id = %s AND color = %s",
                                    (id, color),
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
                                        "overallRating": product[13],
                                    }
                                    recommendResults.append(product_info)
                except mysql.connector.Error as e:
                    print(f"Error: {e}")
                finally:
                    if "connection" in locals() and connection.is_connected():
                        cursor.close()
                        connection.close()
                        print("MySQL connection is closed")
                # If can not find the product by product name
                if not recommendResults:
                    return jsonify(
                        {"error": "No products found for the given product_name"}
                    )
                else:
                    # Declare a dictionary has results and content
                    response_data = {"result": "success", "content": recommendResults}
                    # Make sure "results" apear before "content" in dictionary
                    response_data = {
                        "result": response_data["result"],
                        "content": response_data["content"],
                    }
                    # print(response_data)
                    return jsonify(response_data)
        except Exception as e:
            return jsonify({"error": str(e)})


if __name__ == "__main__":
    # app.run(debug=True) #run on local host
    app.run(host="0.0.0.0", port=5000)
