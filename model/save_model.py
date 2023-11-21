import tensorflow
from keras.layers import GlobalMaxPooling2D
from keras.applications import ResNet50

model = ResNet50(weights='imagenet', include_top=False,
                 input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

model.save('fashion-recommendation-system\\model\\model.pb')
