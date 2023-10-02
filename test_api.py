import requests

# Gửi hình ảnh lên API
files = {'file': (
    'D:\workspace\Recommend_system\\fashion-recommendation-system\\testdata\1163.jpg', open('1163.jpg', 'rb'))}
response = requests.post('http://localhost:5000/recommend', files=files)

if response.status_code == 200:
    recommendations = response.json()
    print("Recommendations:")
    for rec in recommendations:
        print(rec)
else:
    print("Error:", response.json())
